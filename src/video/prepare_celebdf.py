"""Pick a balanced set of Celeb-DF v2 videos (plus the FOMM / FaceVid2Vid / FaceSwap fakes
made from Celeb-DF in 01_Raw_Data) and link them into data/video_subset/<label>/<dataset>/,
ready for normalize_videos.py and extract_faces.py.

Why the grouping: a Celeb-DF fake shows two people (target and source), so linking every
video that shares a person chains almost all 59 people together, and then no fair
train/test split exists. Celeb-DF swaps faces mostly inside small circles of people, so
people are split into groups (graph communities) and only fakes whose two people are in
the same group are used. Each group then gets as many fakes as real videos.

Usage (from the repo root):
    python src/video/prepare_celebdf.py --celebdf <folder with Celeb-real, Celeb-synthesis, YouTube-real>
"""
import os
import re
import csv
import random
import shutil
import argparse
from collections import Counter, defaultdict
import networkx as nx

SEED = 42
SUBSET_DIR = "data/video_subset"
YOUTUBE_REAL = 150          # real YouTube videos to add (they have no fakes in Celeb-DF v2)
GROUP_RESOLUTION = 2.0      # higher = more, smaller groups (2.0 gives 11 groups)
EXTRA_FAKES = {             # dataset name -> folder of extra Celeb-DF-based fakes
    "fomm-cdf": "01_Raw_Data/FOMM/fomm/cdf",
    "facevid2vid-cdf": "01_Raw_Data/FaceVid2Vid/facevid2vid/cdf",
    "faceswap-cdf": "01_Raw_Data/FaceSwap",
}


def people(stem):
    parts = re.sub(r"^CDF_", "", stem).split("_")
    return [p for p in parts if p.startswith("id")]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)        # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)   # different drive: copy instead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celebdf", required=True)
    args = parser.parse_args()
    rng = random.Random(SEED)

    stems = lambda d: sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".mp4"))
    reals = stems(os.path.join(args.celebdf, "Celeb-real"))
    fakes = stems(os.path.join(args.celebdf, "Celeb-synthesis"))

    # Group people so that most face swaps stay inside one group
    graph = nx.Graph()
    for stem in fakes:
        a, b = people(stem)
        w = graph.get_edge_data(a, b, {"weight": 0})["weight"]
        graph.add_edge(a, b, weight=w + 1)
    groups = nx.community.louvain_communities(graph, weight="weight", resolution=GROUP_RESOLUTION, seed=SEED)
    group_of = {p: i for i, g in enumerate(groups) for p in g}
    same_group = lambda stem: len({group_of.get(p) for p in people(stem)}) == 1 and people(stem)[0] in group_of

    chosen = []  # (label, dataset, source path, group)
    for stem in reals:
        chosen.append(("real", "cdf", os.path.join(args.celebdf, "Celeb-real", stem + ".mp4"), group_of.get(people(stem)[0], -1)))
    extra_by_group = Counter()
    for dataset, folder in EXTRA_FAKES.items():
        for stem in stems(folder):
            if same_group(stem):
                g = group_of[people(stem)[0]]
                chosen.append(("fake", dataset, os.path.join(folder, stem + ".mp4"), g))
                extra_by_group[g] += 1

    # Celeb-DF fakes: in each group, top up to as many fakes as real videos
    real_by_group = Counter(group_of.get(people(s)[0], -1) for s in reals)
    candidates = defaultdict(list)
    for stem in fakes:
        if same_group(stem):
            candidates[group_of[people(stem)[0]]].append(stem)
    for g, pool in sorted(candidates.items()):
        need = max(real_by_group[g] - extra_by_group[g], 0)
        for stem in rng.sample(pool, min(need, len(pool))):
            chosen.append(("fake", "cdf", os.path.join(args.celebdf, "Celeb-synthesis", stem + ".mp4"), g))

    youtube = stems(os.path.join(args.celebdf, "YouTube-real"))
    for stem in rng.sample(youtube, min(YOUTUBE_REAL, len(youtube))):
        chosen.append(("real", "cdf-youtube", os.path.join(args.celebdf, "YouTube-real", stem + ".mp4"), -1))

    rows = []
    for label, dataset, src, g in chosen:
        dst = os.path.join(SUBSET_DIR, label, dataset, os.path.basename(src))
        link(src, dst)
        rows.append({"label": label, "dataset": dataset, "video": os.path.splitext(os.path.basename(src))[0], "group": g})
    with open(os.path.join(SUBSET_DIR, "celebdf_selection.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "dataset", "video", "group"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(groups)} people groups; linked {len(rows)} videos into {SUBSET_DIR}")
    for key, n in sorted(Counter((r["label"], r["dataset"]) for r in rows).items()):
        print(f"  {key[0]:4} {key[1]:16} {n}")


if __name__ == "__main__":
    main()
