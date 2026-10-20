"""Shared data loading with a proper train / validation / test split.

- train: 85% of the original `train` folder (used to fit the model)
- val:   15% of the original `train` folder (used for checkpoint selection,
         early stopping and LR scheduling)
- test:  the original `valid` folder (NEVER used for any training decision;
         only evaluate.py touches it)

The split is stratified (same real/fake ratio in train and val) and seeded, and
the chosen indices are saved to notebooks/split_indices.json so every run (and
every teammate) uses exactly the same split.
"""
import os
import json
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = "data/dataset/rvf10k"
SPLIT_FILE = "notebooks/split_indices.json"
VAL_FRACTION = 0.15
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def stratified_split(targets, val_fraction=VAL_FRACTION, seed=SEED):
    """Return (train_idx, val_idx), keeping the class ratio in both parts."""
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for t in sorted(by_class):
        idxs = by_class[t][:]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return sorted(train_idx), sorted(val_idx)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_tf, eval_tf


def _load_or_create_split(targets):
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            saved = json.load(f)
        if saved["n_samples"] == len(targets):
            return saved["train_idx"], saved["val_idx"]
        print("Saved split does not match the dataset size - creating a new one.")

    train_idx, val_idx = stratified_split(targets)
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump({"seed": SEED, "val_fraction": VAL_FRACTION,
                   "n_samples": len(targets),
                   "train_idx": train_idx, "val_idx": val_idx}, f)
    return train_idx, val_idx


def get_loaders(batch_size=32, num_workers=0):
    """Return train_loader, val_loader, test_loader, class_names."""
    train_tf, eval_tf = get_transforms()

    # Two views of the same folder: one with augmentation, one without.
    full_train_aug = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    full_train_plain = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=eval_tf)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=eval_tf)

    train_idx, val_idx = _load_or_create_split(full_train_aug.targets)

    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_plain, val_idx)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, full_train_aug.classes
