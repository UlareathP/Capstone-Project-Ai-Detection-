"""Train a segment-level ResNet18 on log-mel spectrograms, selecting the best epoch by
clip-level validation accuracy (mean P(fake) over each clip's segments). Same settings
as the video model (src/video/train.py): backbone, unfrozen layer4, learning rates,
LR schedule and early stopping. Class balance comes from an attack-balanced sampler.

Usage (from the repo root):
    python src/audio/train.py
"""
import os
import sys
import csv
import random
from collections import Counter
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, SEED, SPLITS_PATH, SegmentDataset, accuracy,
                    build_model, device, eval_transform, make_splits, predict,
                    read_manifest, train_transform)

BATCH_SIZE = 32
EPOCHS = 30
LR_HEAD = 0.001
LR_BACKBONE = 0.0001
PATIENCE = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Using device:", device)

clips = read_manifest()
present = {c["label"] for c in clips}
if present != set(CLASS_NAMES):
    sys.exit(f"Need both real and fake clips, but the manifest only has: {sorted(present)}.")

clips = make_splits(clips)
with open(SPLITS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(clips[0].keys()))
    writer.writeheader()
    writer.writerows(clips)

by_split = {s: [c for c in clips if c["split"] == s] for s in ("train", "valid", "test")}
for s, cs in by_split.items():
    counts = {k: sum(c["label"] == k for c in cs) for k in CLASS_NAMES}
    print(f"{s}: {len(cs)} clips {counts}")

train_dataset = SegmentDataset(by_split["train"], train_transform)
valid_dataset = SegmentDataset(by_split["valid"], eval_transform)
print(f"train segments: {len(train_dataset)}, valid segments: {len(valid_dataset)}")

# Balanced sampling: half of each batch is real, and the fake half is shared equally
# between attack types, so rare attacks (A09-A16 have ~10 clips each) aren't drowned out
# by common ones (~125 each). This replaces the class-weighted loss used for video.
groups = [by_split["train"][ci]["attack_id"] for _, _, ci in train_dataset.samples]
group_size = Counter(groups)
n_attacks = len([g for g in group_size if g != "bonafide"])
weights = [0.5 / group_size[g] if g == "bonafide" else 0.5 / (n_attacks * group_size[g]) for g in groups]
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"attack types in training: {n_attacks}")

model = build_model(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LR_BACKBONE},
    {"params": model.fc.parameters(), "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

best_val_acc, best_seg_acc, epochs_no_improve = -1.0, 0.0, 0
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for specs, labels, _ in train_loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(specs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    (sl, sp), (cl, cp) = predict(model, valid_loader, len(by_split["valid"]))
    seg_acc, clip_acc = accuracy(sl, sp), accuracy(cl, cp)
    scheduler.step(clip_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} "
          f"- Val segment acc: {seg_acc:.4f} - Val clip acc: {clip_acc:.4f}", flush=True)

    if clip_acc > best_val_acc:
        best_val_acc, best_seg_acc, epochs_no_improve = clip_acc, seg_acc, 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  -> New best model saved (val clip acc={clip_acc:.4f})", flush=True)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

print(f"\nBest validation clip accuracy: {best_val_acc:.4f} (segment acc {best_seg_acc:.4f})")

log_path = "notebooks/audio_experiment_log.csv"
log_exists = os.path.exists(log_path)
os.makedirs("notebooks", exist_ok=True)
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "datasets", "test_dataset", "config", "best_val_clip_acc", "best_val_segment_acc"])
    writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        "+".join(sorted({c["dataset"] for c in clips})),
        "official_asvspoof5_eval_partition",
        "resnet18_layer4_unfrozen+specmask+quality_degrade+mp3_64k_clips+silence_trim+mean_clip_prob"
        "+pooled_train_dev_speaker_split+attack_balanced_sampler",
        f"{best_val_acc:.4f}",
        f"{best_seg_acc:.4f}",
    ])
print(f"Logged result to {log_path}. Run src/audio/evaluate.py for test-set results.")
