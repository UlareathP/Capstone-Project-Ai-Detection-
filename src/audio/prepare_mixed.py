"""Build a training set that mixes clean FoR (for-norm) with re-recorded FoR (for-rerec),
in the layout src/audio expects (data/audio_raw/<split>/ + data/audio_meta/*.csv).

Each source keeps its OWN official split, so no clip is shared between train and test:
  train       <- training/   folders of both versions
  validation  <- validation/ folders of both versions
  test        <- testing/    folders of both versions (never trained on)

File names carry the source ("norm_real_000001", "rerec_fake_000001") so eval_mixed.py can
score the two test sets separately. Fakes get attack_id "spoof_norm" / "spoof_rerec", so
the training sampler gives both sources equal weight.

Usage (from the repo root):
    python src/audio/prepare_mixed.py --norm "<...>/for-norm/for-norm" --rerec "<...>/for-rerec/for-rerecorded"
"""
import os
import csv
import random
import shutil
import argparse

SEED = 42
FOLDERS = {"train": "training", "validation": "validation", "test": "testing"}
# clips per class, per source, per split
LIMITS = {
    "norm": {"train": 2000, "validation": 300, "test": 600},
    "rerec": {"train": 1500, "validation": 300, "test": 600},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", required=True, help="for-norm folder holding training/, validation/, testing/")
    parser.add_argument("--rerec", required=True, help="for-rerec folder holding training/, validation/, testing/")
    parser.add_argument("--raw", default="data/audio_raw")
    parser.add_argument("--meta", default="data/audio_meta")
    args = parser.parse_args()
    sources = {"norm": args.norm, "rerec": args.rerec}

    for src, root in sources.items():
        for folder in FOLDERS.values():
            for label in ("real", "fake"):
                p = os.path.join(root, folder, label)
                if not os.path.isdir(p):
                    raise SystemExit(f"Missing folder: {p}")

    rng = random.Random(SEED)
    os.makedirs(args.meta, exist_ok=True)

    for split, folder in FOLDERS.items():
        out_dir = os.path.join(args.raw, split)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for src, root in sources.items():
            for label, csv_label in (("real", "bonafide"), ("fake", "spoof")):
                src_dir = os.path.join(root, folder, label)
                names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(".wav"))
                rng.shuffle(names)
                names = sorted(names[:LIMITS[src][split]])
                for i, name in enumerate(names, 1):
                    file_id = f"{src}_{label}_{i:06d}"
                    shutil.copyfile(os.path.join(src_dir, name), os.path.join(out_dir, file_id + ".wav"))
                    rows.append({"file_id": file_id, "label": csv_label, "speaker_id": file_id,
                                 "attack_id": "bonafide" if label == "real" else f"spoof_{src}",
                                 "codec": "none"})
                print(f"[{split}] {src} {label}: {len(names)} clips")
        with open(os.path.join(args.meta, f"{split}_subset_final.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_id", "label", "speaker_id", "attack_id", "codec"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Clips are in {args.raw}, labels in {args.meta}.")


if __name__ == "__main__":
    main()
