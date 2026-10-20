"""Shared settings for the image pipeline. Mirrors src/video/common.py so both
models see faces the same way: same face crop, same augmentation (including
RandomQualityDegrade), same model, same P(fake) scoring."""
import io
import os
import csv
import random
from collections import defaultdict
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms, models

FACES_DIR = "data/image_faces"
MANIFEST_PATH = os.path.join(FACES_DIR, "manifest.csv")
SPLITS_PATH = os.path.join(FACES_DIR, "splits.csv")
MODEL_PATH = "models/image_resnet18_best.pt"
CLASS_NAMES = ["fake", "real"]  # alphabetical, same order as the video pipeline
SEED = 42
VAL_FRACTION = 0.15  # share of the original train folder used for validation

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class RandomQualityDegrade:
    """Randomly blur, downscale or JPEG-compress a face crop. Applied to real AND fake
    faces alike, so image quality stops being a clue and the model must learn the
    manipulation itself. Identical to the video pipeline."""

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
        if random.random() < 0.5:
            w, h = img.size
            s = random.uniform(0.3, 0.8)
            img = img.resize((max(int(w * s), 16), max(int(h * s), 16)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if random.random() < 0.7:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(20, 75))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


train_transform = transforms.Compose([
    RandomQualityDegrade(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_manifest():
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def stratified_val(names_by_class, val_fraction=VAL_FRACTION, seed=SEED):
    """Same split logic as src/data.py (the earlier image pipeline): shuffle each class
    with a seeded RNG and take the first 15% as validation. Given the same sorted file
    lists it picks exactly the same validation images, so results stay comparable.
    Returns a set of (class, name) pairs."""
    rng = random.Random(seed)
    val = set()
    for cls in sorted(names_by_class):
        names = names_by_class[cls][:]
        rng.shuffle(names)
        val.update((cls, n) for n in names[:int(round(len(names) * val_fraction))])
    return val


def make_splits(images):
    """Assign each image to train/valid/test. The dataset's own `valid` folder is the
    held-out test set; 15% of its `train` folder becomes validation. The split is drawn
    over every image (with or without a detected face), so it doesn't shift when the
    face detector misses one."""
    original_train = defaultdict(list)
    for im in sorted(images, key=lambda im: im["image"]):
        if im["source_split"] == "train":
            original_train[im["label"]].append(im["image"])
    val = stratified_val(original_train)
    out = []
    for im in images:
        if im["source_split"] == "valid":
            split = "test"
        elif (im["label"], im["image"]) in val:
            split = "valid"
        else:
            split = "train"
        out.append({**im, "split": split})
    return out


def read_splits():
    with open(SPLITS_PATH, newline="") as f:
        return list(csv.DictReader(f))


class FaceDataset(Dataset):
    """One sample per face crop; also returns the image index, like the video FrameDataset."""

    def __init__(self, images, transform):
        self.samples = [(os.path.join(FACES_DIR, im["label"], im["image_id"] + ".jpg"),
                         CLASS_NAMES.index(im["label"]), i) for i, im in enumerate(images)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, i = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, i


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4") or name.startswith("fc")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def predict(model, loader):
    """Returns (labels, fake_probs) for every image in the loader."""
    model.eval()
    labels_out, probs_out = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 0]  # P(fake)
            probs_out.extend(probs.cpu().tolist())
            labels_out.extend(labels.tolist())
    return labels_out, probs_out


def accuracy(labels, fake_probs):
    preds = [0 if p >= 0.5 else 1 for p in fake_probs]
    return sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
