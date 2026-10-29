"""Decode each clip, trim leading/trailing silence, and save evenly spaced 2-second
segments. The audio counterpart of src/video/extract_faces.py: segments play the role
of face crops, and the app uses the same sample_segments() function.

Silence is trimmed because in ASVspoof data the length of the silence around the speech
differs between real and fake clips; a model could cheat by measuring it.

Usage (from the repo root):
    python src/audio/extract_segments.py --src data/audio_normalized --meta data/audio_meta
    python src/audio/extract_segments.py --src data/audio_normalized/scenefake --folders

With --meta (ASVspoof 5): --src holds train/, validation/ and test/ folders of clips and
--meta holds the <split>_subset_final.csv label files (bonafide = real, spoof = fake).
With --folders (e.g. SceneFake): --src holds <train|dev|eval>/<real|fake>/ folders, and the
folder name of --src is used as the dataset name.

Each run replaces only its own dataset's rows in the manifest, so datasets can be added
one at a time.
"""
import os
import csv
import argparse
import subprocess
import numpy as np
import imageio_ffmpeg

SEGMENTS_DIR = "data/audio_segments"
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 2 * SAMPLE_RATE
SEGMENTS_PER_CLIP = 6        # evenly spaced, so long and short clips count equally
SILENCE_DB = 40.0            # frames this far below the loudest frame count as silence
DATASET = "asvspoof5"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode(path):
    """Any audio/video file -> float32 mono waveform at 16 kHz (None if unreadable)."""
    cmd = [FFMPEG, "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def trim_silence(wave, frame=400, hop=160):
    """Cut leading and trailing frames that are SILENCE_DB quieter than the loudest frame."""
    if len(wave) < frame:
        return wave
    n = 1 + (len(wave) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms_db = 10 * np.log10(np.mean(wave[idx] ** 2, axis=1) + 1e-10)
    voiced = np.where(rms_db > rms_db.max() - SILENCE_DB)[0]
    return wave[voiced[0] * hop: voiced[-1] * hop + frame]


def sample_segments(path, n_segments=SEGMENTS_PER_CLIP):
    """Yield (start_sample, total_samples, segment) for evenly spaced 2-second segments of
    the speech in a clip. Clips shorter than one segment are zero-padded."""
    wave = decode(path)
    if wave is None or len(wave) == 0:
        return
    wave = trim_silence(wave)
    total = len(wave)
    if total <= SEGMENT_SAMPLES:
        yield 0, total, np.pad(wave, (0, SEGMENT_SAMPLES - total))
        return
    starts = sorted({int(round(s)) for s in np.linspace(0, total - SEGMENT_SAMPLES, n_segments)})
    for s in starts:
        yield s, total, wave[s:s + SEGMENT_SAMPLES]


def process_clip(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for start, _, seg in sample_segments(path):
        pcm = np.clip(seg * 32768.0, -32768, 32767).astype(np.int16)  # half the disk space of float32
        np.save(os.path.join(out_dir, f"seg_{start:07d}.npy"), pcm)
        saved += 1
    return saved


MANIFEST_FIELDS = ["clip_id", "dataset", "clip", "label", "source_split",
                   "speaker_id", "attack_id", "codec", "n_segments"]
FOLDER_SPLITS = {"train": "train", "dev": "validation", "eval": "test"}


def rows_from_asvspoof(src, meta_dir):
    rows = []
    for source_split in ("train", "validation", "test"):
        with open(os.path.join(meta_dir, f"{source_split}_subset_final.csv"), newline="") as f:
            meta = list(csv.DictReader(f))
        folder = os.path.join(src, source_split)
        files = {os.path.splitext(n)[0]: n for n in os.listdir(folder)} if os.path.isdir(folder) else {}
        missing = 0
        for i, m in enumerate(meta, 1):
            label = "real" if m["label"] == "bonafide" else "fake"
            name = files.get(m["file_id"])
            if name is None:
                missing += 1
                continue
            clip_id = f"{DATASET}__{m['file_id']}"
            n = process_clip(os.path.join(folder, name), os.path.join(SEGMENTS_DIR, label, clip_id))
            if n:
                rows.append({"clip_id": clip_id, "dataset": DATASET, "clip": m["file_id"], "label": label,
                             "source_split": source_split, "speaker_id": m["speaker_id"], "attack_id": m["attack_id"],
                             "codec": m["codec"], "n_segments": n})
            else:
                print(f"  !! no audio decoded, skipping {name}")
            if i % 250 == 0 or i == len(meta):
                print(f"[{source_split}] {i}/{len(meta)} clips", flush=True)
        if missing:
            print(f"  !! {missing} {source_split} clips listed in the metadata were not found in {folder}")
    return rows


def rows_from_folders(src, dataset):
    """<train|dev|eval>/<real|fake>/ folders. File names like SceneFake's A_123_5_C.wav:
    the serial (123) groups a clip with its other versions so they stay in one split, and
    the last letter (the scene-swapping method) is used as the attack type."""
    rows = []
    for folder_split, source_split in FOLDER_SPLITS.items():
        for label in ("fake", "real"):
            folder = os.path.join(src, folder_split, label)
            if not os.path.isdir(folder):
                continue
            names = sorted(os.listdir(folder))
            for i, name in enumerate(names, 1):
                stem = os.path.splitext(name)[0]
                parts = stem.split("_")
                serial = "_".join(parts[:2])
                attack = "bonafide" if label == "real" else f"{dataset}_{parts[-1] if len(parts) >= 4 else 'fake'}"
                clip_id = f"{dataset}__{stem}"
                n = process_clip(os.path.join(folder, name), os.path.join(SEGMENTS_DIR, label, clip_id))
                if n:
                    rows.append({"clip_id": clip_id, "dataset": dataset, "clip": stem, "label": label,
                                 "source_split": source_split, "speaker_id": f"{dataset}_{serial}",
                                 "attack_id": attack, "codec": "-", "n_segments": n})
                else:
                    print(f"  !! no audio decoded, skipping {name}")
                if i % 250 == 0 or i == len(names):
                    print(f"[{folder_split}/{label}] {i}/{len(names)} clips", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder of clips (see the usage notes above)")
    parser.add_argument("--meta", help="ASVspoof 5: folder with <split>_subset_final.csv files")
    parser.add_argument("--folders", action="store_true", help="--src has <split>/<label>/ folders")
    args = parser.parse_args()
    if bool(args.meta) == args.folders:
        parser.error("use either --meta (ASVspoof 5) or --folders")

    if args.folders:
        dataset = os.path.basename(os.path.normpath(args.src))
        rows = rows_from_folders(args.src, dataset)
    else:
        dataset = DATASET
        rows = rows_from_asvspoof(args.src, args.meta)

    # Keep the other datasets already in the manifest; replace only this dataset's rows
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    manifest = os.path.join(SEGMENTS_DIR, "manifest.csv")
    others = []
    if os.path.exists(manifest):
        with open(manifest, newline="") as f:
            others = [r for r in csv.DictReader(f) if r["dataset"] != dataset]
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(others + rows)

    for label in ("fake", "real"):
        sel = [r for r in rows if r["label"] == label]
        print(f"{label}: {len(sel)} clips, {sum(r['n_segments'] for r in sel)} segments")
    print(f"Wrote {SEGMENTS_DIR}/manifest.csv")


if __name__ == "__main__":
    main()
