"""Evaluate the trained video model on the held-out test split (never seen during
training or model selection). Reports both frame-level and video-level metrics.

Usage (from the repo root):
    python src/video/evaluate.py
"""
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, FrameDataset, build_model, device,
                    eval_transform, predict, read_splits)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_PATH, help="model file to evaluate")
args = parser.parse_args()
MODEL_PATH = args.model
print("Evaluating", MODEL_PATH)

test_videos = [v for v in read_splits() if v["split"] == "test"]
test_dataset = FrameDataset(test_videos, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test set: {len(test_videos)} videos, {len(test_dataset)} frames")

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

(fl, fp), (vl, vp) = predict(model, test_loader, len(test_videos))
os.makedirs("notebooks", exist_ok=True)

lines = []
for level, labels, probs in (("frame", fl, fp), ("video", vl, vp)):
    preds = [0 if p >= 0.5 else 1 for p in probs]
    report = classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
    # roc_auc_score expects scores for the positive class (label 1 = real)
    auc = roc_auc_score(labels, [1 - p for p in probs]) if len(set(labels)) == 2 else float("nan")
    lines.append(f"=== {level}-level (n={len(labels)}) ===\n{report}ROC AUC: {auc:.4f}\n")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Video Deepfake Detection - {level.title()}-level")
    plt.tight_layout()
    plt.savefig(f"notebooks/video_confusion_matrix_{level}.png")
    plt.close(fig)

# Video-level accuracy per dataset and class (predict() keeps videos in test_videos order)
lines.append("=== video-level accuracy per dataset and class ===")
video_preds = [0 if p >= 0.5 else 1 for p in vp]
for ds in sorted({v["dataset"] for v in test_videos}):
    for k, cls in enumerate(CLASS_NAMES):
        idx = [i for i, v in enumerate(test_videos) if v["dataset"] == ds and vl[i] == k]
        if idx:
            right = sum(video_preds[i] == k for i in idx)
            lines.append(f"{ds:14} {cls:5} {right}/{len(idx)} correct ({right / len(idx):.1%})")
lines.append("")

text = "\n".join(lines)
print(text)
with open("notebooks/video_classification_report.txt", "w") as f:
    f.write(text)
print("Saved notebooks/video_classification_report.txt and video_confusion_matrix_{frame,video}.png")
