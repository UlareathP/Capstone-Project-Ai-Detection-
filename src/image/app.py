"""Local web UI: upload an image, get a real/fake verdict from the trained model.

Runs the same steps as the video app (normalize -> crop face -> score) and returns
the same JSON fields, so one front end can show image and video results side by side.

Usage (from the repo root):
    python src/image/app.py
then open http://127.0.0.1:5001  (the video app uses port 5000)
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
from extract_faces import face_from_file
from normalize_images import normalize

MAX_UPLOAD_MB = 25
FAKE_THRESHOLD = 0.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
_model, _model_mtime = None, None


def get_model():
    # Loaded lazily and reloaded when the checkpoint changes, so the app can run while training
    global _model, _model_mtime
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found - run src/image/train.py first.")
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
    start = time.time()
    # Re-save exactly like the training images so compression matches what the model saw
    fd, encoded = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        if not normalize(path, encoded):
            return {"error": "This file could not be read as an image. Try a JPG, PNG or WEBP file."}
        frames = score_faces(model, encoded)
    finally:
        os.remove(encoded)
    if not frames:
        return {"error": "No face was found in this image. The model needs a clearly visible face."}

    p_fake = sum(f["p_fake"] for f in frames) / len(frames)
    is_fake = p_fake >= FAKE_THRESHOLD
    # Same fields as the video app; an image is treated as a one-frame video
    return {
        "media": "image",
        "verdict": "fake" if is_fake else "real",
        "p_fake": round(p_fake, 4),
        "confidence": round(p_fake if is_fake else 1 - p_fake, 4),
        "faces_found": len(frames),
        "frames_sampled": 1,
        "fake_frames": sum(f["p_fake"] >= FAKE_THRESHOLD for f in frames),
        "elapsed_sec": round(time.time() - start, 1),
        "model_updated": model_updated(),
        "frames": frames,
    }


def score_faces(model, path):
    face = face_from_file(path)
    if face is None:
        return []
    img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        probs = torch.softmax(model(eval_transform(img).unsqueeze(0).to(device)), dim=1)[0]
    return [{
        "frame": 0,
        "position": 0.0,
        "p_fake": round(probs[CLASS_NAMES.index("fake")].item(), 4),
        "thumb": thumbnail(face),
    }]


@app.route("/")
def index():
    return render_template("index.html", model_updated=model_updated(), max_mb=MAX_UPLOAD_MB)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "No image uploaded."}), 400
    suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
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
    app.run(host="127.0.0.1", port=5001, debug=False)
