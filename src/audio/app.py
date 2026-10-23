"""Local web UI: upload an audio clip, get a real/fake verdict from the trained model.

Runs the same steps as the video app (normalize -> sample segments -> score each ->
average) and returns the same JSON fields, so one front end can show image, video
and audio results side by side. Each 2-second segment plays the role of a video frame.

Usage (from the repo root):
    python src/audio/app.py
then open http://127.0.0.1:5002  (video uses 5000, image uses 5001)
"""
import os
import sys
import time
import base64
import tempfile
from datetime import datetime
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, SAMPLE_RATE, build_model, device,
                    spectrogram01, to_model_input)
from extract_segments import SEGMENTS_PER_CLIP, sample_segments
from normalize_audio import normalize

MAX_UPLOAD_MB = 50
FAKE_THRESHOLD = 0.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
_model, _model_mtime = None, None


def get_model():
    # Loaded lazily and reloaded when the checkpoint changes, so the app can run while training
    global _model, _model_mtime
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found - run src/audio/train.py first.")
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


def thumbnail(spec01):
    """Spectrogram as a small colour image (low frequencies at the bottom)."""
    img = (np.flipud(spec01.numpy()) * 255).astype(np.uint8)
    img = cv2.applyColorMap(cv2.resize(img, (112, 112)), cv2.COLORMAP_MAGMA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def analyse(path):
    model = get_model()
    start = time.time()
    # Re-encode exactly like the training clips so compression matches what the model saw
    fd, encoded = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        if not normalize(path, encoded):
            return {"error": "This file could not be read as audio. Try a WAV, MP3, M4A, OGG or FLAC file."}
        frames = score_segments(model, encoded)
    finally:
        os.remove(encoded)
    if not frames:
        return {"error": "No audio could be decoded from this file."}

    p_fake = sum(f["p_fake"] for f in frames) / len(frames)
    is_fake = p_fake >= FAKE_THRESHOLD
    # Same fields as the video app; "frames" are the scored 2-second segments
    return {
        "media": "audio",
        "verdict": "fake" if is_fake else "real",
        "p_fake": round(p_fake, 4),
        "confidence": round(p_fake if is_fake else 1 - p_fake, 4),
        "faces_found": len(frames),
        "frames_sampled": SEGMENTS_PER_CLIP,
        "fake_frames": sum(f["p_fake"] >= FAKE_THRESHOLD for f in frames),
        "elapsed_sec": round(time.time() - start, 1),
        "model_updated": model_updated(),
        "frames": frames,
    }


def score_segments(model, path):
    frames = []
    with torch.no_grad():
        for seg_start, total, seg in sample_segments(path):
            spec = spectrogram01(seg)
            probs = torch.softmax(model(to_model_input(spec).unsqueeze(0).to(device)), dim=1)[0]
            frames.append({
                "frame": round(seg_start / SAMPLE_RATE, 1),   # segment start, in seconds
                "position": seg_start / max(total, 1),
                "p_fake": round(probs[CLASS_NAMES.index("fake")].item(), 4),
                "thumb": thumbnail(spec),
            })
    return frames


@app.route("/")
def index():
    return render_template("index.html", model_updated=model_updated(),
                           n_frames=SEGMENTS_PER_CLIP, max_mb=MAX_UPLOAD_MB)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "No audio uploaded."}), 400
    suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
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
    app.run(host="127.0.0.1", port=5002, debug=False)
