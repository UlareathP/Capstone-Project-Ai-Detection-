"""Evaluate the trained image model on the held-out test split (the dataset's own
`valid` folder, never seen during training or model selection). Same report format
as src/video/evaluate.py: classification report, ROC AUC and a confusion matrix.

Usage (from the repo root):
    python src/image/evaluate.py
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
from common import (CLASS_NAMES, MODEL_PATH, FaceDataset, build_model, device,
                    eval_transform, predict, read_splits)

test_images = [im for im in read_splits() if im["split"] == "test" and int(im["n_faces"])]
test_dataset = FaceDataset(test_images, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test set: {len(test_images)} images")

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

labels, probs = predict(model, test_loader)
os.makedirs("notebooks", exist_ok=True)

preds = [0 if p >= 0.5 else 1 for p in probs]
report = classification_report(labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
# roc_auc_score expects scores for the positive class (label 1 = real)
auc = roc_auc_score(labels, [1 - p for p in probs]) if len(set(labels)) == 2 else float("nan")
text = f"=== image-level (n={len(labels)}) ===\n{report}ROC AUC: {auc:.4f}\n"
text += "\n=== accuracy per dataset and class ===\n"
for ds in sorted({im["dataset"] for im in test_images}):
    for c, cls in enumerate(CLASS_NAMES):
        idx = [i for i, im in enumerate(test_images) if im["dataset"] == ds and labels[i] == c]
        if idx:
            right = sum(preds[i] == c for i in idx)
            text += f"{ds:18} {cls:5} {right}/{len(idx)} correct ({right / len(idx):.1%})\n"

cm = confusion_matrix(labels, preds, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Image Deepfake Detection - Image-level")
plt.tight_layout()
plt.savefig("notebooks/image_confusion_matrix.png")
plt.close(fig)

print(text)
with open("notebooks/image_classification_report.txt", "w") as f:
    f.write(text)
print("Saved notebooks/image_classification_report.txt and image_confusion_matrix.png")
