"""Copy a balanced random sample of SceneFake clips out of the Kaggle zip, without
unzipping all 6 GB. SceneFake is ~10x larger than our ASVspoof 5 subset, so using all
of it would drown out the AI-voice attacks (and take very long to train on a CPU).

Fake clips are spread evenly over the 4 speech-enhancement methods used to swap the
acoustic scene (file name A_<serial>_<snr>_<method>.wav, method A=fullsubnet, B=mmse,
C=spec, D=wiener).

Usage (from the repo root):
    python src/audio/sample_scenefake.py --zip 01_Raw_Data/SceneFake/scenefake.zip
Writes data/audio_raw/scenefake/<train|dev|eval>/<real|fake>/*.wav
"""
import os
import random
import zipfile
import argparse
from collections import defaultdict

PER_CLASS = {"train": 500, "dev": 100, "eval": 200}  # clips per label in each split
SEED = 42


def method_of(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    parts = stem.split("_")
    return parts[3] if len(parts) >= 4 else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--dst", default="data/audio_raw/scenefake")
    args = parser.parse_args()

    rng = random.Random(SEED)
    with zipfile.ZipFile(args.zip) as z:
        # Group wav files by (split, label); the zip may have an extra top-level folder
        groups = defaultdict(list)
        for name in z.namelist():
            parts = name.replace("\\", "/").split("/")
            if not name.lower().endswith(".wav") or len(parts) < 3:
                continue
            split, label = parts[-3], parts[-2]
            if split in PER_CLASS and label in ("real", "fake"):
                groups[(split, label)].append(name)

        for (split, label), names in sorted(groups.items()):
            names.sort()
            rng.shuffle(names)
            n = PER_CLASS[split]
            if label == "fake":  # equal share per enhancement method
                by_method = defaultdict(list)
                for name in names:
                    by_method[method_of(name)].append(name)
                per = n // len(by_method)
                chosen = [x for m in sorted(by_method) for x in by_method[m][:per]]
            else:
                chosen = names[:n]
            out_dir = os.path.join(args.dst, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for name in chosen:
                with z.open(name) as src, open(os.path.join(out_dir, os.path.basename(name)), "wb") as dst:
                    dst.write(src.read())
            print(f"{split}/{label}: copied {len(chosen)} of {len(names)} clips")


if __name__ == "__main__":
    main()
