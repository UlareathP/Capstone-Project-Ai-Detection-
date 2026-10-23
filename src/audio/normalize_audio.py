"""Re-encode every clip with the same settings (16 kHz mono MP3 at 64 kbps), so file
format and compression can't separate real from fake. The audio counterpart of
src/video/normalize_videos.py.

Usage (from the repo root):
    python src/audio/normalize_audio.py --src data/audio_raw --dst data/audio_normalized
Mirrors the folder structure of --src into --dst.
"""
import os
import argparse
import subprocess
import imageio_ffmpeg

SAMPLE_RATE = 16000
BITRATE = "64k"     # same bitrate for every clip
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def normalize(src_path, dst_path):
    """Re-encode one clip; returns True on success."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src_path, "-vn",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", BITRATE, dst_path]
    return subprocess.run(cmd).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    done = failed = 0
    for root, _, files in os.walk(args.src):
        for name in sorted(files):
            if not name.lower().endswith(AUDIO_EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, name), args.src)
            out = os.path.join(args.dst, os.path.splitext(rel)[0] + ".mp3")
            if normalize(os.path.join(root, name), out):
                done += 1
            else:
                failed += 1
                print("  !! failed:", rel)
    print(f"Re-encoded {done} clips ({SAMPLE_RATE} Hz mono MP3 {BITRATE}) into {args.dst}"
          + (f", {failed} failed" if failed else ""))


if __name__ == "__main__":
    main()
