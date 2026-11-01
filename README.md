# Multimodal Deepfake Detection

UC ITS Capstone, Project 2026-S2-72: **Multimodal AI Framework for Robust Detection of Deepfake and Synthetic Media**.

One web app that checks an **image**, a **video** or an **audio** file and returns a real-or-fake verdict with a confidence score. Each media type has its own fine-tuned ResNet-18 model.

**Live app:** https://gokalpyldz-deepfake-detector.hf.space/

**Quick links:** [User guide](docs/USER_GUIDE.md) | [Setup](#setup) | [Results](#results-held-out-test-data) | [Known limitations](#known-limitations)

## What each pipeline does

| Media | Steps | Final verdict |
|---|---|---|
| Image | Re-save in the training format, crop the largest face with YuNet, resize to 224 x 224, score with ResNet-18 | Fake probability of the face |
| Video | Sample 32 frames, crop the face in each frame, score every face | Mean fake probability over the frames |
| Audio | Trim silence, take six 2-second segments, convert each to a log-mel spectrogram, score every segment | Mean fake probability over the segments |

The web page also shows the per-frame (video) and per-segment (audio) fake scores.

## Results (held-out test data)

| Media | Test size | Accuracy |
|---|---|---|
| Image | 3,010 images | 90.9% |
| Video | 18 videos (small test) | 88.9% |
| Audio | 800 clips | 83.0% |

Details are in `notebooks/*_classification_report.txt`, `notebooks/*_experiment_log.csv` and `docs/`.

## Setup

1. Install Python 3 and create a virtual environment:

   ```
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the trained model files from the [GitHub release v1.0-models](https://github.com/UlareathP/Capstone-Project-Ai-Detection-/releases/tag/v1.0-models) and copy them into the `models/` folder (they are not stored in Git because of their size):

   - `models/image_resnet18_best.pt`
   - `models/video_resnet18_best.pt`
   - `models/audio_resnet18_best.pt`

3. From the repository root, start the app:

   ```
   python src/api/server.py
   ```

4. Open http://localhost:8000, choose the Image, Video or Audio tab, upload a file and press **Analyse**.

Run all commands from the repository root, because the model paths are relative.
The first image or video run downloads the YuNet face detector (needs internet).

## Training and evaluation

Each media type has its own folder with the same scripts:

```
python src/<image|video|audio>/train.py
python src/<image|video|audio>/evaluate.py
```

Data preparation scripts (`extract_faces.py`, `extract_segments.py`, `normalize_*.py`) are in the same folders. Seeds are fixed (42) and the splits are saved (`models/baseline/splits.csv`, `notebooks/split_indices.json`) so results can be repeated.

## Data

All datasets are public. We used subsets that fit on student laptops.

- Image: rvf10k (Kaggle) and Generated Photos
- Video: DF40, DeepFaceLab face-swap subset
- Audio: ASVspoof 5 and SceneFake

## Known limitations

- Image and video need a visible face.
- Video was tested on only 18 videos and one face-swap method.
- Audio catches far fewer ASVspoof 5 fakes (about 43.5%) than SceneFake fakes (about 97%), and was trained on English speech.
- Heavy compression, noise or poor lighting make results less reliable.
- Results are guidance, not proof. Do not use them as evidence for important decisions.
- Uploaded files are deleted straight after analysis, but no authentication or privacy notice is built in.

## Project structure

```
src/api/        Web server and page (all three tabs)
src/image/      Image pipeline
src/video/      Video pipeline
src/audio/      Audio pipeline
src/gradcam.py  Grad-CAM script (extra, not part of the app)
notebooks/      Experiment logs, classification reports, split files
models/         Model files (not in Git) and split files
docs/           Image test report and confusion matrix
```
