"""Convert the Fake-or-Real (FoR) dataset into the layout src/audio expects.

FoR is organised as <version>/<split>/{real,fake}/*.wav and has no speaker or attack
labels, so this script
  * copies a random, class-balanced subset of clips flat into data/audio_raw/<split>/
    under unique names (real_000001.wav, fake_000001.wav), and
  * writes data/audio_meta/<split>_subset_final.csv with the columns the pipeline reads
    (file_id, label, speaker_id, attack_id, codec).

Every clip gets its own speaker_id (unknown in FoR), and all fakes share attack_id
"spoof". src/audio/train.py then works unchanged. FoR's own "testing" folder becomes
the test set.

Usage (from the repo root):
    python src/audio/prepare_for.py --src "C:/Programming/PRML Project/PRML-Project2/Audio Data/for-norm/for-norm"
Then:
    python src/audio/normalize_audio.py --src data/audio_raw --dst data/audio_normalized
    python src/audio/extract_segments.py --src data/audio_normalized --meta data/audio_meta
    python src/audio/train.py
    python src/audio/evaluate.py
"""
import os
import csv
import random
import shutil
import argparse

SEED = 42
# FoR folder name -> folder/CSV name the pipeline expects
SPLITS = {"training": "train", "validation": "validation", "testing": "test"}


def list_wavs(folder):
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(".wav"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="folder containing training/, validation/ and testing/ (e.g. .../for-norm/for-norm)")
    parser.add_argument("--raw", default="data/audio_raw")
    parser.add_argument("--meta", default="data/audio_meta")
    parser.add_argument("--train-per-class", type=int, default=3000)
    parser.add_argument("--val-per-class", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    args = parser.parse_args()
    limits = {"train": args.train_per_class, "validation": args.val_per_class, "test": args.test_per_class}

    for folder in SPLITS:
        for label in ("real", "fake"):
            p = os.path.join(args.src, folder, label)
            if not os.path.isdir(p):
                raise SystemExit(f"Missing folder: {p}\n--src should contain training/, validation/ and testing/.")

    rng = random.Random(SEED)
    os.makedirs(args.meta, exist_ok=True)

    for folder, split in SPLITS.items():
        out_dir = os.path.join(args.raw, split)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for label, csv_label in (("real", "bonafide"), ("fake", "spoof")):
            src_dir = os.path.join(args.src, folder, label)
            names = list_wavs(src_dir)
            rng.shuffle(names)
            names = names[:limits[split]]
            for i, name in enumerate(sorted(names), 1):
                file_id = f"{label}_{i:06d}"
                shutil.copyfile(os.path.join(src_dir, name), os.path.join(out_dir, file_id + ".wav"))
                rows.append({"file_id": file_id, "label": csv_label,
                             "speaker_id": file_id,  # speaker unknown in FoR: one group per clip
                             "attack_id": "bonafide" if label == "real" else "spoof",
                             "codec": "none"})
            print(f"[{split}] {label}: {len(names)} clips")
        with open(os.path.join(args.meta, f"{split}_subset_final.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_id", "label", "speaker_id", "attack_id", "codec"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Clips are in {args.raw}, labels in {args.meta}.")


if __name__ == "__main__":
    main()
