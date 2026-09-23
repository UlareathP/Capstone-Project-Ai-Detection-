import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

DATA_DIR = "data/dataset/rvf10k"
MODEL_PATH = "models/resnet18_deepfake_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
class_names = valid_dataset.classes  # ['fake', 'real']

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# --- Confusion matrix plot ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Deepfake Detection - Confusion Matrix")
plt.tight_layout()
os.makedirs("notebooks", exist_ok=True)
plt.savefig("notebooks/confusion_matrix.png")
print("Saved confusion matrix to notebooks/confusion_matrix.png")

# --- Precision / recall / F1 report ---
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)
with open("notebooks/classification_report.txt", "w") as f:
    f.write(report)
print("Saved detailed report to notebooks/classification_report.txt")