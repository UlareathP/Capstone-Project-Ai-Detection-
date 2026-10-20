"""Final evaluation on the held-out TEST set (the original `valid` folder).

Run this only after training is finished. Do not use these numbers to choose
checkpoints or tune hyperparameters - that is what the validation split is for.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from data import get_loaders

MODEL_PATH = "models/resnet18_deepfake_best.pt"
OUT_DIR = "docs"  # not ignored by .gitignore, so figures can be committed

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

_, _, test_loader, class_names = get_loaders(batch_size=32)  # ['fake', 'real']

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

os.makedirs(OUT_DIR, exist_ok=True)

# --- Confusion matrix plot ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Deepfake Detection - Confusion Matrix (TEST set)")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "test_confusion_matrix.png")
plt.savefig(cm_path)
print(f"Saved confusion matrix to {cm_path}")

# --- Precision / recall / F1 report ---
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)
report_path = os.path.join(OUT_DIR, "test_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Saved detailed report to {report_path}")
