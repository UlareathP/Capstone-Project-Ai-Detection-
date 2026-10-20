import os
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from data import DATA_DIR, VAL_FRACTION, get_loaders

BATCH_SIZE = 32
EPOCHS = 15
LR_HEAD = 0.001       # learning rate for the new final layer
LR_BACKBONE = 0.0001  # smaller LR for the unfrozen backbone layer
PATIENCE = 3          # early stopping: stop if no improvement for N epochs
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Train / validation come from the original train folder. The test set is NOT loaded here.
train_loader, val_loader, _, class_names = get_loaders(BATCH_SIZE)
print("Classes:", class_names)  # ['fake', 'real'] -> index 0 = fake, 1 = real

# --- Model: unfreeze layer4 + fc, freeze everything before it ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name, param in model.named_parameters():
    param.requires_grad = name.startswith("layer4") or name.startswith("fc")

model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: fake (0), real (1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LR_BACKBONE},
    {"params": model.fc.parameters(), "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


best_val_acc = 0.0
epochs_no_improve = 0
os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

log_path = "notebooks/experiment_log.csv"
log_exists = os.path.exists(log_path)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = evaluate(val_loader)
    scheduler.step(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "models/resnet18_deepfake_best.pt")
        print(f"  -> New best model saved (val_acc={val_acc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# --- Log the result to a simple CSV so you can track experiments over time ---
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "dataset", "config", "best_val_accuracy"])
    writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        DATA_DIR,
        f"resnet18_layer4_unfrozen+augmentation | val={int(VAL_FRACTION*100)}% of train, seed={SEED}",
        f"{best_val_acc:.4f}",
    ])
print(f"Logged result to {log_path}")
print("Now run `python src/evaluate.py` to measure performance on the held-out TEST set.")
