"""Merge several datasets prepared in the src/audio layout into ONE dataset, so a single
model can be trained on all of them (e.g. FoR clean + FoR re-recorded + ASVspoof 2019 LA).

Each --src gives: <raw folder> <meta folder> <prefix>. The raw folder holds train/, validation/
and test/ sub-folders; the meta folder holds the three *_subset_final.csv files. The prefix is
added to every file name so ids never collide and eval_all.py can tell the sources apart
(use "-" for no prefix, for files that already carry one, such as norm_ / rerec_).
"""
import os
import csv
import shutil
import argparse

SPLITS = ("train", "validation", "test")
HEADER = ["file_id", "label", "speaker_id", "attack_id", "codec"]
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs=3, action="append", required=True,
                        metavar=("RAW", "META", "PREFIX"))
    parser.add_argument("--raw", default="data/audio_raw")
    parser.add_argument("--meta", default="data/audio_meta")
    args = parser.parse_args()

    os.makedirs(args.meta, exist_ok=True)
    for split in SPLITS:
        out_dir = os.path.join(args.raw, split)
        os.makedirs(out_dir, exist_ok=True)
        rows, seen = [], set()
        for raw, meta, prefix in args.src:
            prefix = "" if prefix == "-" else prefix
            csv_path = os.path.join(meta, f"{split}_subset_final.csv")
            if not os.path.exists(csv_path):
                raise SystemExit(f"Missing {csv_path}")
            files = {os.path.splitext(n)[0]: n for n in os.listdir(os.path.join(raw, split))
                     if n.lower().endswith(AUDIO_EXTS)}
            with open(csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r["file_id"] not in files:
                        raise SystemExit(f"{split}: {r['file_id']} listed in {csv_path} but no audio in {raw}/{split}")
                    new_id = prefix + r["file_id"]
                    if new_id in seen:
                        raise SystemExit(f"Duplicate id {new_id} - give the sources different prefixes.")
                    seen.add(new_id)
                    ext = os.path.splitext(files[r["file_id"]])[1]
                    shutil.copyfile(os.path.join(raw, split, files[r["file_id"]]),
                                    os.path.join(out_dir, new_id + ext))
                    rows.append({**r, "file_id": new_id})
        with open(os.path.join(args.meta, f"{split}_subset_final.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        real = sum(r["label"] == "bonafide" for r in rows)
        print(f"[{split}] {len(rows)} clips ({real} real, {len(rows) - real} fake)")
    print(f"\nDone. Merged data is in {args.raw} and {args.meta}.")


if __name__ == "__main__":
    main()
