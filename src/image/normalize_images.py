"""Re-save every image as JPEG with the same quality, so file format and compression
can't separate real from fake. The image counterpart of src/video/normalize_videos.py
(which re-encodes every video with the same H.264 setting).

Usage (from the repo root):
    python src/image/normalize_images.py --src data/dataset/rvf10k --dst data/image_normalized/rvf10k
Mirrors the folder structure of --src into --dst.
"""
import os
import argparse
from PIL import Image, ImageOps

JPEG_QUALITY = 75   # same quality for every image (lower = more compressed)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def normalize(src_path, dst_path):
    """Re-save one image as RGB JPEG; returns True on success."""
    try:
        img = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")  # respect phone rotation
    except Exception:
        return False
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    img.save(dst_path, format="JPEG", quality=JPEG_QUALITY)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    done = failed = 0
    for root, _, files in os.walk(args.src):
        for name in sorted(files):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, name), args.src)
            out = os.path.join(args.dst, os.path.splitext(rel)[0] + ".jpg")
            if normalize(os.path.join(root, name), out):
                done += 1
            else:
                failed += 1
                print("  !! failed:", rel)
    print(f"Re-saved {done} images (JPEG quality {JPEG_QUALITY}) into {args.dst}" + (f", {failed} failed" if failed else ""))


if __name__ == "__main__":
    main()
