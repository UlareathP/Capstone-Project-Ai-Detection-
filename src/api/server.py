"""One API server for the mobile app: image, video and audio detection in one place.

It loads the existing pipelines (src/image/app.py, src/video/app.py, src/audio/app.py)
and calls their analyse() functions, so results are identical to the web apps.
It listens on all network interfaces so a phone on the same Wi-Fi can reach it.

Usage (from the repo root):
    python src/api/server.py
then point the mobile app at http://<this-computer's-IP>:8000
"""
import os
import sys
import socket
import tempfile
import importlib.util
from flask import Flask, jsonify, request

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8000
MAX_UPLOAD_MB = 200

# The three pipelines use the same helper module names (common, extract_faces, ...),
# so each one is loaded with a clean slate and keeps its own copies.
SHARED_NAMES = ("common", "extract_faces", "extract_segments",
                "normalize_images", "normalize_videos", "normalize_audio")


def load_pipeline(kind):
    folder = os.path.join(SRC, kind)
    for name in SHARED_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(f"{kind}_pipeline", os.path.join(folder, "app.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:  # a missing dependency in one pipeline shouldn't stop the others
        return None, str(e)
    finally:
        sys.path.remove(folder)
        for name in SHARED_NAMES:
            sys.modules.pop(name, None)


PIPELINES = {}
for kind in ("image", "video", "audio"):
    module, error = load_pipeline(kind)
    PIPELINES[kind] = {"module": module, "load_error": error}
    print(f"[{kind}] " + ("loaded" if module else f"failed to load: {error}"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def status(kind):
    p = PIPELINES[kind]
    if p["module"] is None:
        return {"available": False, "reason": p["load_error"]}
    if not os.path.exists(p["module"].MODEL_PATH):
        return {"available": False, "reason": f"{p['module'].MODEL_PATH} not found - train this model first."}
    return {"available": True, "model_updated": p["module"].model_updated()}


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "pipelines": {k: status(k) for k in PIPELINES}})


@app.route("/api/predict/<kind>", methods=["POST"])
def predict(kind):
    if kind not in PIPELINES:
        return jsonify({"error": f"Unknown media type '{kind}'."}), 404
    state = status(kind)
    if not state["available"]:
        return jsonify({"error": state["reason"]}), 503

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400
    suffix = os.path.splitext(file.filename)[1].lower() or {"image": ".jpg", "video": ".mp4", "audio": ".m4a"}[kind]
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        file.save(tmp)
        result = PIPELINES[kind]["module"].analyse(tmp)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    finally:
        os.remove(tmp)
    result.setdefault("media", kind)
    return jsonify(result), (422 if "error" in result else 200)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"That file is larger than {MAX_UPLOAD_MB} MB."}), 413


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; this just picks the Wi-Fi interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    print(f"\nIn the mobile app, set the server to:  http://{lan_ip()}:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
