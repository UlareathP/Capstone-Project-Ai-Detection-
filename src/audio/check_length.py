"""Accuracy on the test set split by clip length (1 segment = short, 6 = long).
If 'fake long' and 'real short' stay near 1.000, the model is not just reading length.

Run from the repo root:  python src/audio/check_length.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from common import (MODEL_PATH, SegmentDataset, build_model, device,
                    eval_transform, predict, read_splits)

for path in (MODEL_PATH, "data/audio_segments/splits.csv"):
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run this from the repo root, after training, "
                 f"and before deleting data/audio_segments or renaming the model.")

test = [c for c in read_splits() if c["split"] == "test"]
model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loader = DataLoader(SegmentDataset(test, eval_transform), batch_size=32, shuffle=False)
_, (labels, probs) = predict(model, loader, len(test))

hits = defaultdict(lambda: [0, 0])
for c, p in zip(test, probs):
    pred = "fake" if p >= 0.5 else "real"
    key = (c["label"], "short" if int(c["n_segments"]) == 1 else "long")
    hits[key][0] += pred == c["label"]
    hits[key][1] += 1

for key, (ok, n) in sorted(hits.items()):
    print(f"{key[0]:5} {key[1]:5} {ok}/{n} = {ok / n:.3f}")