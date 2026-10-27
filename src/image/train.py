"""Train a ResNet18 on face crops from still images, with the same settings as the
video model (src/video/train.py): same backbone, unfrozen layer4, learning rates,
augmentation, class-weighted loss, LR schedule and early stopping.

Usage (from the repo root):
    python src/image/train.py
"""
import os
import sys
import csv
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, SEED, SPLITS_PATH, FaceDataset, accuracy,
                    build_model, device, eval_transform, make_splits, predict,
                    read_manifest, train_transform)

BATCH_SIZE = 32
EPOCHS = 15
LR_HEAD = 0.001
LR_BACKBONE = 0.0001
PATIENCE = 3
# Small extra datasets are repeated in the training set, so a handful of images
# isn't drowned out by thousands of rvf10k images (augmentation varies each copy)
OVERSAMPLE = {"generated_photos": 25}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Using device:", device)

images = read_manifest()
present = {im["label"] for im in images}
if present != set(CLASS_NAMES):
    sys.exit(f"Need both real and fake images, but the manifest only has: {sorted(present)}.")

images = make_splits(images)
with open(SPLITS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(images[0].keys()))
    writer.writeheader()
    writer.writerows(images)

with_face = [im for im in images if int(im["n_faces"])]
by_split = {s: [im for im in with_face if im["split"] == s] for s in ("train", "valid", "test")}
for s, ims in by_split.items():
    counts = {c: sum(im["label"] == c for im in ims) for c in CLASS_NAMES}
    print(f"{s}: {len(ims)} images {counts}")
print(f"skipped (no face found): {len(images) - len(with_face)}")

train_images = [im for im in by_split["train"] for _ in range(OVERSAMPLE.get(im["dataset"], 1))]
train_dataset = FaceDataset(train_images, train_transform)
valid_dataset = FaceDataset(by_split["valid"], eval_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Weight the loss so an unbalanced real/fake count doesn't bias the model
counts = torch.bincount(torch.tensor([s[1] for s in train_dataset.samples]), minlength=2).float()
class_weights = (counts.sum() / (2 * counts)).to(device)

model = build_model(pretrained=True)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LR_BACKBONE},
    {"params": model.fc.parameters(), "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

best_val_acc, epochs_no_improve = -1.0, 0
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images_batch, labels, _ in train_loader:
        images_batch, labels = images_batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images_batch), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    vl, vp = predict(model, valid_loader)
    val_acc = accuracy(vl, vp)
    scheduler.step(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Val image acc: {val_acc:.4f}", flush=True)

    if val_acc > best_val_acc:
        best_val_acc, epochs_no_improve = val_acc, 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  -> New best model saved (val image acc={val_acc:.4f})", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

print(f"\nBest validation image accuracy: {best_val_acc:.4f}")

log_path = "notebooks/image_experiment_log.csv"
log_exists = os.path.exists(log_path)
os.makedirs("notebooks", exist_ok=True)
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "datasets", "test_dataset", "config", "best_val_image_acc"])
    writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        "+".join(sorted({im["dataset"] for im in images})),
        "original_valid_folder",
        "resnet18_layer4_unfrozen+augmentation+quality_degrade+jpeg75_images+yunet_face_crop"
        + "".join(f"+oversample_{ds}x{n}" for ds, n in OVERSAMPLE.items()),
        f"{best_val_acc:.4f}",
    ])
print(f"Logged result to {log_path}. Run src/image/evaluate.py for test-set results.")
