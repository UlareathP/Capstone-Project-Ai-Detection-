"""Re-encode every video with the same H.264 settings so compression level can't
separate real from fake (in the raw data, fakes have ~half the bitrate of reals).

Usage (from the repo root):
    python src/video/normalize_videos.py --src data/video_subset --dst data/video_normalized
Mirrors the folder structure of --src into --dst. Videos already re-encoded are skipped,
so an interrupted run can be resumed.
"""
import os
import argparse
import subprocess
import imageio_ffmpeg

CRF = 30          # same constant quality for every video (higher = more compressed)
PRESET = "medium"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def normalize(src_path, dst_path):
    """Re-encode one video; returns True on success."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src_path,
           "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
           "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # x264 needs even sizes
           "-an", dst_path]
    return subprocess.run(cmd).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    done = failed = 0
    for root, _, files in os.walk(args.src):
        for name in sorted(files):
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, name), args.src)
            out = os.path.join(args.dst, os.path.splitext(rel)[0] + ".mp4")
            if os.path.exists(out) and os.path.getsize(out) > 0:  # done in an earlier run
                done += 1
                continue
            if normalize(os.path.join(root, name), out):
                done += 1
            else:
                failed += 1
                print("  !! failed:", rel)
    print(f"Re-encoded {done} videos (CRF {CRF}) into {args.dst}" + (f", {failed} failed" if failed else ""))


if __name__ == "__main__":
    main()
