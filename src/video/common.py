import io
import os
import csv
import random
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold

FRAMES_DIR = "data/video_frames"
MANIFEST_PATH = os.path.join(FRAMES_DIR, "manifest.csv")
SPLITS_PATH = os.path.join(FRAMES_DIR, "splits.csv")
MODEL_PATH = "models/video_resnet18_best.pt"
CLASS_NAMES = ["fake", "real"]  # alphabetical, same order as the image pipeline
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class RandomQualityDegrade:
    """Randomly blur, downscale or JPEG-compress a face crop. Applied to real AND fake
    faces alike, so image quality stops being a clue and the model must learn the
    manipulation itself (the raw fakes are blurrier than the raw reals)."""

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


def identity_of(video_stem):
    # Celeb-DF fakes: "id0_id1_0001" -> target "id0"; reals: "id0_0001" -> "id0"
    # FF++ / DeepFaceLab fakes: "003_000" -> target "003"; reals: "003" -> "003"
    return video_stem.split("_")[0]


def identities_in(video_stem):
    """Every person in a video: a fake contains the target AND the swapped-in source."""
    parts = video_stem.split("_")
    if parts[0].startswith("id"):  # Celeb-DF: last part is a clip number, not a person
        parts = [p for p in parts if p.startswith("id")]
    return parts


def identity_groups(videos):
    """Union-find over people, so videos sharing any person land in the same group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = []
    for v in videos:
        people = [f'{v["dataset"]}:{p}' for p in identities_in(v["video"])]
        for p in people[1:]:
            parent[find(p)] = find(people[0])
        keys.append(people[0])
    return [find(k) for k in keys]


def make_splits(videos, test_dataset=None):
    """Assign each video to train/valid/test, grouping by connected identities so the
    same person never appears in two splits. If test_dataset is given, that whole
    dataset becomes the test set (cross-dataset evaluation)."""
    labels = [v["label"] for v in videos]
    groups = identity_groups(videos)
    split = {}

    if test_dataset:
        pool = [i for i, v in enumerate(videos) if v["dataset"] != test_dataset]
        for i, v in enumerate(videos):
            if v["dataset"] == test_dataset:
                split[i] = "test"
    else:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        rest, test = next(sgkf.split(videos, labels, groups))
        for i in test:
            split[i] = "test"
        pool = list(rest)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(pool, [labels[i] for i in pool], [groups[i] for i in pool]))
    for i in tr:
        split[pool[i]] = "train"
    for i in va:
        split[pool[i]] = "valid"

    return [{**v, "split": split[i]} for i, v in enumerate(videos)]


def read_splits():
    with open(SPLITS_PATH, newline="") as f:
        return list(csv.DictReader(f))


class FrameDataset(Dataset):
    """One sample per face crop; also returns the video index so predictions
    can be averaged per video."""

    def __init__(self, videos, transform):
        self.samples = []
        self.video_ids = [v["video_id"] for v in videos]
        for vi, v in enumerate(videos):
            folder = os.path.join(FRAMES_DIR, v["label"], v["video_id"])
            label = CLASS_NAMES.index(v["label"])
            for name in sorted(os.listdir(folder)):
                self.samples.append((os.path.join(folder, name), label, vi))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, vi = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, vi


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4") or name.startswith("fc")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def predict(model, loader, n_videos):
    """Returns frame-level (labels, fake_probs) and video-level (labels, mean fake_probs)."""
    model.eval()
    frame_labels, frame_probs, frame_videos = [], [], []
    with torch.no_grad():
        for images, labels, vids in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 0]  # P(fake)
            frame_probs.extend(probs.cpu().tolist())
            frame_labels.extend(labels.tolist())
            frame_videos.extend(vids.tolist())

    sums, counts, video_labels = [0.0] * n_videos, [0] * n_videos, [None] * n_videos
    for p, y, v in zip(frame_probs, frame_labels, frame_videos):
        sums[v] += p
        counts[v] += 1
        video_labels[v] = y
    keep = [v for v in range(n_videos) if counts[v]]
    video_probs = [sums[v] / counts[v] for v in keep]
    return (frame_labels, frame_probs), ([video_labels[v] for v in keep], video_probs)


def accuracy(labels, fake_probs):
    preds = [0 if p >= 0.5 else 1 for p in fake_probs]
    return sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
