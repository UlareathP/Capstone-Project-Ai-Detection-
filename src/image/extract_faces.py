"""Crop the face out of every image and save it as a 224x224 crop. Uses exactly the
same detector (YuNet), margin and crop size as src/video/extract_faces.py, so image
and video models are trained on identically prepared faces.

Usage (from the repo root):
    python src/image/extract_faces.py --src data/image_normalized/rvf10k

--src must contain <split>/<label>/ folders (e.g. train/fake, valid/real). The folder
name of --src ("rvf10k") is used as the dataset name.
"""
import os
import csv
import argparse
import urllib.request
import cv2

FACES_DIR = "data/image_faces"
FACE_MARGIN = 0.3       # extra context around the detected face box (same as video)
CROP_SIZE = 224
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

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
    """Largest face in a BGR image, with margin, resized to CROP_SIZE (same as video)."""
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


def face_from_file(path):
    """Read an image file and return its face crop (BGR), or None if no face was found."""
    frame = cv2.imread(path)
    return None if frame is None else crop_face(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder with <split>/<label>/ image folders")
    args = parser.parse_args()
    dataset = os.path.basename(os.path.normpath(args.src))

    rows = []
    for source_split in sorted(os.listdir(args.src)):
        for label in ("fake", "real"):
            folder = os.path.join(args.src, source_split, label)
            if not os.path.isdir(folder):
                continue
            images = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
            os.makedirs(os.path.join(FACES_DIR, label), exist_ok=True)
            missed = 0
            for i, name in enumerate(images, 1):
                stem = os.path.splitext(name)[0]
                image_id = f"{dataset}__{source_split}__{stem}"
                face = face_from_file(os.path.join(folder, name))
                if face is not None:
                    cv2.imwrite(os.path.join(FACES_DIR, label, image_id + ".jpg"), face)
                else:
                    missed += 1
                rows.append({"image_id": image_id, "dataset": dataset, "image": stem, "label": label,
                             "source_split": source_split, "n_faces": int(face is not None)})
                if i % 500 == 0 or i == len(images):
                    print(f"[{label}/{source_split}] {i}/{len(images)} images, {missed} without a face")

    # Keep the other datasets already in the manifest; replace only this dataset's rows
    os.makedirs(FACES_DIR, exist_ok=True)
    manifest = os.path.join(FACES_DIR, "manifest.csv")
    others = []
    if os.path.exists(manifest):
        with open(manifest, newline="") as f:
            others = [r for r in csv.DictReader(f) if r["dataset"] != dataset]
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "dataset", "image", "label", "source_split", "n_faces"])
        writer.writeheader()
        writer.writerows(others + rows)

    for label in ("fake", "real"):
        sel = [r for r in rows if r["label"] == label]
        print(f"{label}: {len(sel)} images, {sum(r['n_faces'] for r in sel)} face crops")
    print(f"Wrote {FACES_DIR}/manifest.csv")


if __name__ == "__main__":
    main()
