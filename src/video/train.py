"""Train a frame-level ResNet18 on face crops, selecting the best epoch by
video-level validation accuracy (mean P(fake) over each video's frames).

Usage (from the repo root):
    python src/video/train.py                     # random split by identity
    python src/video/train.py --test-dataset cdf  # hold out a whole dataset
"""
import os
import sys
import csv
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLASS_NAMES, MODEL_PATH, SEED, SPLITS_PATH, FrameDataset,
                    FramesPerVideoSampler, accuracy, build_model, device, eval_transform,
                    make_splits, predict, read_manifest, train_transform)

BATCH_SIZE = 32
LR_HEAD = 0.001
LR_BACKBONE = 0.0001

parser = argparse.ArgumentParser()
parser.add_argument("--test-dataset", default=None, help="hold out this dataset (e.g. cdf) as the test set")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement")
parser.add_argument("--frames-per-epoch", type=int, default=8,
                    help="random frames per video in each epoch (validation always uses all of them)")
parser.add_argument("--unfreeze", nargs="+", default=["layer4"], choices=["layer2", "layer3", "layer4"],
                    help="ResNet18 blocks to fine-tune (the rest keep their ImageNet weights)")
parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay; > 0 discourages memorising the training videos")
parser.add_argument("--label-smoothing", type=float, default=0.0,
                    help="soften the 0/1 targets (e.g. 0.1) so the model is less over-confident")
parser.add_argument("--out", default=MODEL_PATH, help="where to save the best model")
args = parser.parse_args()
FRAMES_PER_EPOCH, EPOCHS, PATIENCE = args.frames_per_epoch, args.epochs, args.patience
print(f"Fine-tuning {args.unfreeze} + fc, up to {EPOCHS} epochs (patience {PATIENCE}), "
      f"{FRAMES_PER_EPOCH} frames per video per epoch, weight decay {args.weight_decay}, "
      f"label smoothing {args.label_smoothing}, saving to {args.out}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Using device:", device)

videos = read_manifest()
present = {v["label"] for v in videos}
if present != set(CLASS_NAMES):
    sys.exit(f"Need both real and fake videos, but the manifest only has: {sorted(present)}. "
             "Re-run extract_faces.py with --real <folders>.")

videos = make_splits(videos, args.test_dataset)
with open(SPLITS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(videos[0].keys()))
    writer.writeheader()
    writer.writerows(videos)

by_split = {s: [v for v in videos if v["split"] == s] for s in ("train", "valid", "test")}
for s, vs in by_split.items():
    counts = {c: sum(v["label"] == c for v in vs) for c in CLASS_NAMES}
    print(f"{s}: {len(vs)} videos {counts}")

train_dataset = FrameDataset(by_split["train"], train_transform)
valid_dataset = FrameDataset(by_split["valid"], eval_transform)
sampler = FramesPerVideoSampler(train_dataset, FRAMES_PER_EPOCH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"train frames: {len(train_dataset)} ({len(sampler)} used per epoch, "
      f"{FRAMES_PER_EPOCH} per video), valid frames: {len(valid_dataset)}")

# Weight the loss so an unbalanced real/fake video count doesn't bias the model
labels_by_video = {vi: label for _, label, vi in train_dataset.samples}
frame_counts = torch.zeros(2)
for vi, idx in sampler.by_video.items():
    frame_counts[labels_by_video[vi]] += min(FRAMES_PER_EPOCH, len(idx))
class_weights = (frame_counts.sum() / (2 * frame_counts)).to(device)

model = build_model(pretrained=True, trainable=args.unfreeze)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
param_groups = ([{"params": getattr(model, block).parameters(), "lr": LR_BACKBONE} for block in args.unfreeze]
                + [{"params": model.fc.parameters(), "lr": LR_HEAD}])
# Plain Adam when there is no weight decay keeps the original runs reproducible
optimizer = (torch.optim.AdamW(param_groups, weight_decay=args.weight_decay) if args.weight_decay > 0
             else torch.optim.Adam(param_groups))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

best_val_acc, best_frame_acc, epochs_no_improve = -1.0, 0.0, 0
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    (fl, fp), (vl, vp) = predict(model, valid_loader, len(by_split["valid"]))
    frame_acc, video_acc = accuracy(fl, fp), accuracy(vl, vp)
    scheduler.step(video_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} "
          f"- Val frame acc: {frame_acc:.4f} - Val video acc: {video_acc:.4f}", flush=True)

    if video_acc > best_val_acc:
        best_val_acc, best_frame_acc, epochs_no_improve = video_acc, frame_acc, 0
        torch.save(model.state_dict(), args.out)
        print(f"  -> New best model saved (val video acc={video_acc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early.")
            break

print(f"\nBest validation video accuracy: {best_val_acc:.4f} (frame acc {best_frame_acc:.4f})")

log_path = "notebooks/video_experiment_log.csv"
log_exists = os.path.exists(log_path)
os.makedirs("notebooks", exist_ok=True)
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not log_exists:
        writer.writerow(["timestamp", "datasets", "test_dataset", "config", "best_val_video_acc", "best_val_frame_acc"])
    writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        "+".join(sorted({v["dataset"] for v in videos})),
        args.test_dataset or "random_identity_split",
        f"resnet18_{'+'.join(args.unfreeze)}_unfrozen+augmentation+quality_degrade+crf30_videos+mean_video_prob"
        f"+{FRAMES_PER_EPOCH}_frames_per_video_per_epoch+patience{PATIENCE}"
        + (f"+weight_decay{args.weight_decay}" if args.weight_decay else "")
        + (f"+label_smoothing{args.label_smoothing}" if args.label_smoothing else ""),
        f"{best_val_acc:.4f}",
        f"{best_frame_acc:.4f}",
    ])
print(f"Logged result to {log_path}. Run src/video/evaluate.py for test-set results.")
