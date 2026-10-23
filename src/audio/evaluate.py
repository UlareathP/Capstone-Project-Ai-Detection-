"""Evaluate the trained audio model on the held-out test split (ASVspoof 5 evaluation
clips: unseen speakers and unseen attack types). Same report format as
src/video/evaluate.py: segment-level and clip-level metrics, ROC AUC, confusion matrices.

Usage (from the repo root):
    python src/audio/evaluate.py
"""
import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, SegmentDataset, build_model, device,
                    eval_transform, predict, read_splits)

test_clips = [c for c in read_splits() if c["split"] == "test"]
test_dataset = SegmentDataset(test_clips, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test set: {len(test_clips)} clips, {len(test_dataset)} segments")

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

(sl, sp), (cl, cp) = predict(model, test_loader, len(test_clips))
os.makedirs("notebooks", exist_ok=True)

lines = []
for level, labels, probs in (("segment", sl, sp), ("clip", cl, cp)):
    preds = [0 if p >= 0.5 else 1 for p in probs]
    report = classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
    # roc_auc_score expects scores for the positive class (label 1 = real)
    auc = roc_auc_score(labels, [1 - p for p in probs]) if len(set(labels)) == 2 else float("nan")
    lines.append(f"=== {level}-level (n={len(labels)}) ===\n{report}ROC AUC: {auc:.4f}\n")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Audio Deepfake Detection - {level.title()}-level")
    plt.tight_layout()
    plt.savefig(f"notebooks/audio_confusion_matrix_{level}.png")
    plt.close(fig)

text = "\n".join(lines)
print(text)
with open("notebooks/audio_classification_report.txt", "w") as f:
    f.write(text)
print("Saved notebooks/audio_classification_report.txt and audio_confusion_matrix_{segment,clip}.png")
