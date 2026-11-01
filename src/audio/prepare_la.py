"""Build train / validation / test data from an ASVspoof 2019 LA download (PA is ignored).

Point --root at the folder that holds the LA audio folders (e.g. ASVspoof2019_LA_train/flac,
..._dev/flac, ..._eval/flac) and the protocol files (*.cm.train.trn.txt, *.cm.dev.trl.txt,
*.cm.eval.trl.txt). The script finds them itself.

  LA train -> train       LA dev -> validation       LA eval -> test (unseen attacks)

Writes data/audio_raw/<split>/ and data/audio_meta/<split>_subset_final.csv, the layout
src/audio expects. Then run normalize_audio.py, extract_segments.py, train.py, evaluate.py.

Usage (from the repo root):
    python src/audio/prepare_la.py --root "C:/path/to/ASVspoof"
"""
import os
import re
import csv
import random
import shutil
import argparse

AUDIO_EXTS = (".flac", ".wav")
HEADER = ["file_id", "label", "speaker_id", "attack_id", "codec"]
LA_RE = re.compile(r"(^|[_.\-\s])la([_.\-\s]|$)", re.I)
PA_RE = re.compile(r"(^|[_.\-\s])pa([_.\-\s]|$)", re.I)


def split_of(text):
    t = text.lower()
    for word, split in (("eval", "test"), ("train", "train"), ("dev", "validation")):
        if word in t:
            return split
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing the ASVspoof LA download")
    parser.add_argument("--raw", default="data/audio_raw")
    parser.add_argument("--meta", default="data/audio_meta")
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--val-per-class", type=int, default=400)
    parser.add_argument("--test-per-class", type=int, default=1000)
    args = parser.parse_args()
    limits = {"train": args.train_per_class, "validation": args.val_per_class, "test": args.test_per_class}

    audio = {s: {} for s in limits}   # split -> {file stem: path}
    protos = {}                       # split -> protocol path
    for dirpath, _, names in os.walk(args.root):
        parts = dirpath.replace("\\", "/").split("/")
        if any(PA_RE.search(p) for p in parts):
            continue  # PA = replay attacks, not AI speech
        in_la = any(LA_RE.search(p) for p in parts)
        for n in names:
            low = n.lower()
            if low.endswith(AUDIO_EXTS) and in_la:
                sp = next((split_of(p) for p in reversed(parts) if split_of(p)), None)
                if sp:
                    audio[sp][os.path.splitext(n)[0]] = os.path.join(dirpath, n)
            elif low.endswith(".txt") and ".cm." in low and ".pa." not in low and (in_la or ".la." in low):
                sp = split_of(low)
                if sp:
                    protos[sp] = os.path.join(dirpath, n)

    for sp in limits:
        print(f"[{sp}] audio files: {len(audio[sp])} | protocol: {protos.get(sp, 'NOT FOUND')}")
    missing = [sp for sp in limits if not audio[sp] or sp not in protos]
    if missing:
        raise SystemExit(f"\nCould not find LA audio and/or a '.cm.' protocol file for: {missing}.\n"
                         "Check --root points at the folder containing the LA train/dev/eval folders and protocol files.")

    rng = random.Random(42)
    os.makedirs(args.meta, exist_ok=True)
    for sp in limits:
        by_label = {"bonafide": [], "spoof": []}
        with open(protos[sp], encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                tokens = line.split()
                label = next((t.lower() for t in tokens if t.lower() in by_label), None)
                fid = next((t for t in tokens if t in audio[sp]), None)
                if not label or not fid:
                    continue
                attack = next((t for t in tokens if re.fullmatch(r"A\d+", t)), None)
                by_label[label].append({
                    "file_id": fid, "label": label,
                    "speaker_id": tokens[0] if tokens[0] != fid else f"unknown_{i}",
                    "attack_id": "bonafide" if label == "bonafide" else (attack or "spoof"),
                    "codec": "none"})
        out_dir = os.path.join(args.raw, sp)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for label, items in by_label.items():
            rng.shuffle(items)
            rows += items[:limits[sp]]
        for r in rows:
            src = audio[sp][r["file_id"]]
            shutil.copyfile(src, os.path.join(out_dir, r["file_id"] + os.path.splitext(src)[1]))
        with open(os.path.join(args.meta, f"{sp}_subset_final.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        attacks = sorted({r["attack_id"] for r in rows if r["label"] == "spoof"})
        print(f"[{sp}] copied {len(by_label['bonafide'][:limits[sp]])} real + "
              f"{len(by_label['spoof'][:limits[sp]])} fake | attacks: {attacks}")

    print(f"\nDone. Clips are in {args.raw}, labels in {args.meta}.")


if __name__ == "__main__":
    main()
