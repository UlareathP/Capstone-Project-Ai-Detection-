"""Sample frames from each video, crop the face, and save 224x224 crops.

Usage (from the repo root):
    python src/video/extract_faces.py --fake facevid2vid/cdf facevid2vid/ff --real <real_dir> [<real_dir> ...]

Each source folder name ("cdf", "ff", ...) is used as the dataset name, so real and
fake folders for the same dataset should share a name, e.g. data/real_videos/cdf.
"""
import os
import csv
import argparse
import urllib.request
import cv2

FRAMES_DIR = "data/video_frames"
FRAMES_PER_VIDEO = 32   # evenly spaced, so long and short clips count equally
FACE_MARGIN = 0.3       # extra context around the detected face box
CROP_SIZE = 224
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# YuNet face detector (OpenCV 5 dropped the Haar cascades)
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

if not os.path.exists(YUNET_PATH):
    os.makedirs("models", exist_ok=True)
    print("Downloading YuNet face detector...")
    urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), score_threshold=0.7)


def crop_face(frame):
    H, W = frame.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])  # largest face
    m = int(max(w, h) * FACE_MARGIN)
    side = max(w, h) + 2 * m
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x1, y1 = min(x0 + side, W), min(y0 + side, H)
    return cv2.resize(frame[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE))


def sample_faces(path, n_frames=FRAMES_PER_VIDEO):
    """Yield (frame_index, total_frames, face_crop_bgr) for evenly spaced frames that contain a face."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return
    step = max(total / n_frames, 1)
    indices = sorted({int(i * step) for i in range(n_frames) if int(i * step) < total})
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        face = crop_face(frame)
        if face is not None:
            yield idx, total, face
    cap.release()


def process_video(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for idx, _, face in sample_faces(path):
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.jpg"), face)
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", nargs="+", default=[], help="folders of fake videos")
    parser.add_argument("--real", nargs="+", default=[], help="folders of real videos")
    args = parser.parse_args()

    rows = []
    for label, folders in (("fake", args.fake), ("real", args.real)):
        for folder in folders:
            dataset = os.path.basename(os.path.normpath(folder))
            videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
            for i, name in enumerate(videos, 1):
                stem = os.path.splitext(name)[0]
                video_id = f"{dataset}__{stem}"
                n = process_video(os.path.join(folder, name), os.path.join(FRAMES_DIR, label, video_id))
                print(f"[{label}/{dataset}] {i}/{len(videos)} {name}: {n} faces")
                if n:
                    rows.append({"video_id": video_id, "dataset": dataset, "video": stem, "label": label, "n_frames": n})
                else:
                    print(f"  !! no faces found, skipping {name}")

    os.makedirs(FRAMES_DIR, exist_ok=True)
    with open(os.path.join(FRAMES_DIR, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "dataset", "video", "label", "n_frames"])
        writer.writeheader()
        writer.writerows(rows)

    for label in ("fake", "real"):
        sel = [r for r in rows if r["label"] == label]
        print(f"{label}: {len(sel)} videos, {sum(r['n_frames'] for r in sel)} face crops")
    print(f"Wrote {FRAMES_DIR}/manifest.csv")


if __name__ == "__main__":
    main()
