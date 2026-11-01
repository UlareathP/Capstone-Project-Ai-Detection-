"""Score the already-trained audio model on whatever clips are in
data/audio_segments/manifest.csv (all of them, regardless of split), without retraining.
Used to test the for-norm model on a different dataset version (for-rerec).

Run from the repo root:  python src/audio/eval_rerec.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from common import (CLASS_NAMES, MODEL_PATH, SegmentDataset, build_model, device,
                    eval_transform, predict, read_manifest)

if not os.path.exists(MODEL_PATH):
    sys.exit(f"Missing {MODEL_PATH}. Train first (and don't rename the model).")

clips = read_manifest()
print(f"{len(clips)} clips from the manifest: "
      f"{sum(c['label'] == 'fake' for c in clips)} fake, {sum(c['label'] == 'real' for c in clips)} real")

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loader = DataLoader(SegmentDataset(clips, eval_transform), batch_size=32, shuffle=False)
_, (labels, probs) = predict(model, loader, len(clips))

preds = [0 if p >= 0.5 else 1 for p in probs]
print(classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))
print("confusion matrix (rows = true fake/real, cols = predicted fake/real):")
print(confusion_matrix(labels, preds, labels=[0, 1]))
if len(set(labels)) == 2:
    print(f"ROC AUC: {roc_auc_score(labels, [1 - p for p in probs]):.4f}")
