"""Shared settings for the audio pipeline. Mirrors src/video/common.py:
a clip is cut into evenly spaced 2-second segments (the audio "frames"), each
segment becomes a log-mel spectrogram (the audio "face crop"), the same ResNet18
scores every segment, and the clip verdict is the mean P(fake)."""
import os
import csv
import random
from math import gcd
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import models

SEGMENTS_DIR = "data/audio_segments"
MANIFEST_PATH = os.path.join(SEGMENTS_DIR, "manifest.csv")
SPLITS_PATH = os.path.join(SEGMENTS_DIR, "splits.csv")
MODEL_PATH = "models/audio_resnet18_best.pt"
CLASS_NAMES = ["fake", "real"]  # alphabetical, same order as the image and video pipelines
SEED = 42
VAL_FOLDS = 7  # 1 of 7 speaker groups (~14%) of the pooled train+dev clips becomes validation

SAMPLE_RATE = 16000
SEGMENT_SEC = 2.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SEC)
N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS = 512, 400, 160, 128  # 25 ms windows, 10 ms hop
TOP_DB = 80.0

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=None):
    """Triangular mel filters (HTK formula), shape [n_mels, n_fft // 2 + 1]."""
    fmax = fmax or sr / 2
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    points = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = points[i], points[i + 1], points[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return torch.from_numpy(fb)


MEL_FB = _mel_filterbank()
WINDOW = torch.hann_window(WIN_LENGTH)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def spectrogram01(wave):
    """float32 waveform -> log-mel spectrogram scaled to [0, 1], shape [N_MELS, frames].
    Measured relative to its own peak, so loudness is not a clue."""
    x = torch.as_tensor(wave, dtype=torch.float32)
    spec = torch.stft(x, N_FFT, HOP_LENGTH, WIN_LENGTH, window=WINDOW, return_complex=True).abs() ** 2
    db = 10 * torch.log10(MEL_FB @ spec + 1e-10)
    db = torch.clamp(db - db.max(), min=-TOP_DB)
    return (db + TOP_DB) / TOP_DB


def to_model_input(spec01):
    """[N_MELS, frames] in [0, 1] -> 3-channel ImageNet-normalized tensor, like an image."""
    return (spec01.unsqueeze(0).repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD


class RandomQualityDegrade:
    """Randomly band-limit, add noise to, or telephone-compress a segment. Applied to real
    AND fake audio alike, so recording quality stops being a clue and the model must learn
    the synthesis itself. The audio counterpart of the image/video RandomQualityDegrade."""

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, wave):
        if random.random() > self.p:
            return wave
        n = len(wave)
        if random.random() < 0.5:  # narrower bandwidth, like phone or low-bitrate audio
            low = random.choice([6000, 8000, 11000, 12000])
            g = gcd(low, SAMPLE_RATE)
            wave = resample_poly(resample_poly(wave, low // g, SAMPLE_RATE // g), SAMPLE_RATE // g, low // g)
            wave = np.pad(wave, (0, max(n - len(wave), 0)))[:n]
        if random.random() < 0.5:  # background noise at 15-40 dB SNR
            power = float(np.mean(wave ** 2)) + 1e-10
            snr = random.uniform(15, 40)
            wave = wave + np.random.randn(n) * np.sqrt(power / 10 ** (snr / 10))
        if random.random() < 0.3:  # 8-bit mu-law, like a telephone codec
            mu = 255.0
            peak = np.max(np.abs(wave)) + 1e-9
            y = np.sign(wave / peak) * np.log1p(mu * np.abs(wave / peak)) / np.log1p(mu)
            y = np.round((y + 1) / 2 * mu) / mu * 2 - 1
            wave = np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu * peak
        return wave.astype(np.float32)


class SpecMask:
    """Mask one random band of frequencies and one stretch of time (SpecAugment).
    Plays the role of the flips/rotations used for face crops."""

    def __call__(self, spec01):
        spec01 = spec01.clone()
        f = random.randint(0, 15)
        f0 = random.randint(0, spec01.shape[0] - f)
        spec01[f0:f0 + f, :] = 0
        t = random.randint(0, 20)
        t0 = random.randint(0, spec01.shape[1] - t)
        spec01[:, t0:t0 + t] = 0
        return spec01


_degrade, _mask = RandomQualityDegrade(), SpecMask()


def train_transform(wave):
    return to_model_input(_mask(spectrogram01(_degrade(wave))))


def eval_transform(wave):
    return to_model_input(spectrogram01(wave))


def read_manifest():
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def make_splits(clips):
    """The official ASVspoof 5 evaluation partition stays the test set (unseen speakers,
    unseen attacks A17-A32). The official train (attacks A01-A08) and development (A09-A16)
    partitions are pooled, so the model learns from 16 attack types instead of 8, then
    split into train/valid by speaker (no speaker in both), keeping the share of each
    attack type similar in both parts - like the identity-grouped split in the video pipeline.
    Other datasets (e.g. SceneFake) follow the same rule: their eval folder is test, and
    train+dev are pooled and split by group. Each dataset is split on its own, so adding
    one doesn't move any ASVspoof 5 clip between train and valid."""
    split = {i: "test" for i, c in enumerate(clips) if c["source_split"] == "test"}
    for dataset in dict.fromkeys(c["dataset"] for c in clips):  # keeps manifest order
        pool = [i for i, c in enumerate(clips) if c["source_split"] != "test" and c["dataset"] == dataset]
        if not pool:
            continue
        sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=SEED)
        tr, va = next(sgkf.split(pool, [clips[i]["attack_id"] for i in pool], [clips[i]["speaker_id"] for i in pool]))
        for j in tr:
            split[pool[j]] = "train"
        for j in va:
            split[pool[j]] = "valid"
    return [{**c, "split": split[i]} for i, c in enumerate(clips)]


def read_splits():
    with open(SPLITS_PATH, newline="") as f:
        return list(csv.DictReader(f))


class SegmentDataset(Dataset):
    """One sample per segment; also returns the clip index so predictions can be
    averaged per clip (like FrameDataset in the video pipeline)."""

    def __init__(self, clips, transform):
        self.samples = []
        for ci, c in enumerate(clips):
            folder = os.path.join(SEGMENTS_DIR, c["label"], c["clip_id"])
            label = CLASS_NAMES.index(c["label"])
            for name in sorted(os.listdir(folder)):
                self.samples.append((os.path.join(folder, name), label, ci))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, ci = self.samples[idx]
        wave = np.load(path).astype(np.float32) / 32768.0
        return self.transform(wave), label, ci


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4") or name.startswith("fc")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def predict(model, loader, n_clips):
    """Returns segment-level (labels, fake_probs) and clip-level (labels, mean fake_probs)."""
    model.eval()
    seg_labels, seg_probs, seg_clips = [], [], []
    with torch.no_grad():
        for specs, labels, clips in loader:
            probs = torch.softmax(model(specs.to(device)), dim=1)[:, 0]  # P(fake)
            seg_probs.extend(probs.cpu().tolist())
            seg_labels.extend(labels.tolist())
            seg_clips.extend(clips.tolist())

    sums, counts, clip_labels = [0.0] * n_clips, [0] * n_clips, [None] * n_clips
    for p, y, c in zip(seg_probs, seg_labels, seg_clips):
        sums[c] += p
        counts[c] += 1
        clip_labels[c] = y
    keep = [c for c in range(n_clips) if counts[c]]
    clip_probs = [sums[c] / counts[c] for c in keep]
    return (seg_labels, seg_probs), ([clip_labels[c] for c in keep], clip_probs)


def accuracy(labels, fake_probs):
    preds = [0 if p >= 0.5 else 1 for p in fake_probs]
    return sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
