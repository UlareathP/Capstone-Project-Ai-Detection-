"""Score the trained model separately on each source's held-out test clips (norm_ = clean FoR,
rerec_ = re-recorded FoR, la_ = ASVspoof 2019 LA), using the test split written by train.py.
Run after training a model on a dataset merged with merge_datasets.py.

Run from the repo root:  python src/audio/eval_all.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from common import (CLASS_NAMES, MODEL_PATH, SegmentDataset, build_model, device,
                    eval_transform, predict, read_splits)

SOURCES = (("norm_", "clean FoR"), ("rerec_", "re-recorded FoR"), ("la_", "ASVspoof 2019 LA (unseen attacks)"))

for path in (MODEL_PATH, "data/audio_segments/splits.csv"):
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run src/audio/train.py first, from the repo root.")

test = [c for c in read_splits() if c["split"] == "test"]
model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

for prefix, name in SOURCES:
    clips = [c for c in test if c["clip"].startswith(prefix)]
    if not clips:
        print(f"\n=== {name}: no test clips found ===")
        continue
    loader = DataLoader(SegmentDataset(clips, eval_transform), batch_size=32, shuffle=False)
    _, (labels, probs) = predict(model, loader, len(clips))
    preds = [0 if p >= 0.5 else 1 for p in probs]
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
    print(f"\n=== {name}: {len(labels)} clips, accuracy {acc:.1%} ===")
    print(classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))
    print("confusion matrix (rows = true fake/real, cols = predicted fake/real):")
    print(confusion_matrix(labels, preds, labels=[0, 1]))
    if len(set(labels)) == 2:
        print(f"ROC AUC: {roc_auc_score(labels, [1 - p for p in probs]):.4f}")
