# Deepfake Detection System

An AI system that detects deepfake and AI-generated **images, videos and audio**, built as part
of the UC ITS Capstone project. Each detector turns its input into a form a ResNet18 network can
score (face crops for images and video, log-mel spectrograms for audio) and returns a
real / fake verdict with a confidence. One web server runs all three.

## What is built

| Detector | Input | Method | Folder |
|---|---|---|---|
| Image | photo | face crop, ResNet18 | `src/image/` |
| Video | video clip | face crops from frames, ResNet18, scores combined per video | `src/video/` |
| Audio | speech clip | 2-second log-mel spectrogram segments, ResNet18, scores averaged per clip | `src/audio/` |
| Combined server | any of the above | one web page with Image / Video / Audio tabs, plus a JSON API for the mobile app | `src/api/server.py` |

Explainability: `src/gradcam.py` produces Grad-CAM heat maps for the image model, and the audio
app shows the spectrogram of every segment it scores.

## Results

All numbers are on held-out test data that the model never trained on.

| Detector | Test set | Result |
|---|---|---|
| Image | rvf10k, 3,010 images | 91% accuracy, ROC AUC 0.969 |
| Video | 18 videos (570 frames) | 89% video-level accuracy, AUC 0.926 (79% at frame level) |
| Audio | clean Fake-or-Real, 1,200 clips | 100% |
| Audio | re-recorded Fake-or-Real, 816 clips | 100% (a model trained without re-recorded audio scored 90%) |
| Audio | ASVspoof 2019 LA, 2,000 clips with attack types not seen in training | 78.6% accuracy, ROC AUC 0.88 |

Detailed reports are in `notebooks/`.

## Known limitations

- **Audio generalisation is the main weakness.** On ASVspoof 2019 LA, the attack types in the test
  set are mostly different from those in training. The model catches about 64% of those fakes and
  almost none of some attack types (for example A10: 9-16% caught in two training runs).
  Training on 25% more data did not change this (80% to 79%).
- **The Fake-or-Real scores are not proof of general deepfake detection.** That benchmark is easy
  (real and fake clips come from different sources), and the same model drops to about 79% on a
  harder test.
- Each model works best on the kind of data it was trained on. Phone recordings or modern
  commercial voices were not part of any test set.
- The video test set is small (18 videos), so its figures carry wide uncertainty.
- Novel generators that none of the training sets contain are not covered.

## Setup

Python 3.12 was used.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Model files

Trained models are not stored in git because of their size (about 45 MB each). Download them
from **[add link here, e.g. the v1.0-models release]** and place them in `models/`:

| File | Used by |
|---|---|
| `image_resnet18_best.pt` | image detector |
| `video_resnet18_best.pt` | video detector |
| `audio_resnet18_best.pt` | audio detector (the combined model trained on Fake-or-Real and ASVspoof 2019 LA) |

## Running

From the repository root:

```powershell
python src/api/server.py        # all three detectors: http://127.0.0.1:8000
```

The server listens on all network interfaces, so a phone on the same Wi-Fi can use
`http://<this-computer's-IP>:8000`. Health check: `/api/health`. Prediction endpoint:
`POST /api/predict/<kind>`, with `image`, `video` or `audio` as the kind.

Each detector can also run on its own:

| Command | Address |
|---|---|
| `python src/video/app.py` | http://127.0.0.1:5000 |
| `python src/image/app.py` | http://127.0.0.1:5001 |
| `python src/audio/app.py` | http://127.0.0.1:5002 |

## Training the audio detector

The datasets are too large for git. Download them separately:

- **Fake-or-Real (FoR)**, versions `for-norm` and `for-rerec`
- **ASVspoof 2019 LA** (the logical-access part, not PA)

Then, from the repository root:

```powershell
python src/audio/prepare_mixed.py --norm <for-norm folder> --rerec <for-rerec folder> --raw data/part_for_raw --meta data/part_for_meta
python src/audio/prepare_la.py --root <ASVspoof folder> --raw data/part_la_raw --meta data/part_la_meta --train-per-class 2000 --val-per-class 300 --test-per-class 1000
python src/audio/merge_datasets.py --src data/part_for_raw data/part_for_meta - --src data/part_la_raw data/part_la_meta la_
python src/audio/normalize_audio.py --src data/audio_raw --dst data/audio_normalized
python src/audio/extract_segments.py --src data/audio_normalized --meta data/audio_meta
python src/audio/train.py
python src/audio/eval_all.py          # test each dataset separately
python src/audio/check_attacks.py     # per-attack breakdown on ASVspoof
```

`train.py` writes `models/audio_resnet18_best.pt`, overwriting the previous file, so copy
models you want to keep. The other scripts in `src/audio/` reproduce earlier experiments
(`prepare_for.py`, `eval_rerec.py`, `check_length.py`). See `notebooks/audio_*.txt` for all
audio results and `notebooks/audio_experiment_log.csv` for the run history.

Training for the image and video detectors lives in `src/image/train.py` and
`src/video/train.py`.

## Repository layout

```
src/
  api/        combined web server and JSON API
  image/      image detector (preprocess, train, evaluate, app)
  video/      video detector
  audio/      audio detector
  gradcam.py  Grad-CAM visualisation
notebooks/    results, reports and experiment logs
models/       trained .pt files (not in git)
data/         datasets and processed data (not in git)
```

## Goals

- Detect previously unseen deepfake generation techniques (partly met: see the audio
  limitations above)
- Maintain accuracy under real-world conditions such as compression, noise and re-recording
  (met for audio re-recording; each model is trained with quality degradation)

## Authors

[add team members' names, and the licence if one applies]
