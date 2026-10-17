"""Local web UI: upload a video, get a real/fake verdict from the trained model.

Usage (from the repo root):
    python src/video/app.py
then open http://127.0.0.1:5000
"""
import os
import sys
import time
import base64
import tempfile
from datetime import datetime
import cv2
import torch
from PIL import Image
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))
from common import CLASS_NAMES, MODEL_PATH, build_model, device, eval_transform
from extract_faces import sample_faces

FRAMES_TO_ANALYSE = 32
MAX_UPLOAD_MB = 200
FAKE_THRESHOLD = 0.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
_model, _model_mtime = None, None


def get_model():
    # Loaded lazily and reloaded when the checkpoint changes, so the app can run while training
    global _model, _model_mtime
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found - run src/video/train.py first.")
    mtime = os.path.getmtime(MODEL_PATH)
    if _model is None or mtime != _model_mtime:
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        _model, _model_mtime = model, mtime
    return _model


def model_updated():
    if not os.path.exists(MODEL_PATH):
        return None
    return datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime("%d %b %Y, %H:%M")


def thumbnail(face_bgr):
    small = cv2.resize(face_bgr, (112, 112))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def analyse(path):
    model = get_model()
    cap = cv2.VideoCapture(path)
    readable = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()
    if not readable:
        return {"error": "This file could not be read as a video. Try an MP4, AVI, MOV or MKV file."}

    start = time.time()
    frames = []
    with torch.no_grad():
        for idx, total, face in sample_faces(path, FRAMES_TO_ANALYSE):
            img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            probs = torch.softmax(model(eval_transform(img).unsqueeze(0).to(device)), dim=1)[0]
            frames.append({
                "frame": idx,
                "position": idx / max(total, 1),
                "p_fake": round(probs[CLASS_NAMES.index("fake")].item(), 4),
                "thumb": thumbnail(face),
            })
    if not frames:
        return {"error": "No face was found in the sampled frames. The model needs a clearly visible face."}

    p_fake = sum(f["p_fake"] for f in frames) / len(frames)
    is_fake = p_fake >= FAKE_THRESHOLD
    return {
        "verdict": "fake" if is_fake else "real",
        "p_fake": round(p_fake, 4),
        "confidence": round(p_fake if is_fake else 1 - p_fake, 4),
        "faces_found": len(frames),
        "frames_sampled": FRAMES_TO_ANALYSE,
        "fake_frames": sum(f["p_fake"] >= FAKE_THRESHOLD for f in frames),
        "elapsed_sec": round(time.time() - start, 1),
        "model_updated": model_updated(),
        "frames": frames,
    }


@app.route("/")
def index():
    return render_template("index.html", model_updated=model_updated(),
                           n_frames=FRAMES_TO_ANALYSE, max_mb=MAX_UPLOAD_MB)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"error": "No video uploaded."}), 400
    suffix = os.path.splitext(file.filename)[1].lower() or ".mp4"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file.save(tmp)
        result = analyse(tmp)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        os.remove(tmp)
    return jsonify(result), (422 if "error" in result else 200)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"That file is larger than {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
