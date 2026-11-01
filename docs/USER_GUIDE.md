# User Guide

For the sponsor and anyone trying the Multimodal Deepfake Detection app.

**Live app:** https://gokalpyldz-deepfake-detector.hf.space/

## 1. Open the app

Open the link above. The first load can take a little longer if the app has been idle. The page has three tabs: **Image**, **Video** and **Audio**. A green "model ready" label at the top shows that the models are loaded.

## 2. Check a file

| Tab | What to upload | What the app does |
|---|---|---|
| Image | A photo with a clearly visible face | Finds the largest face, crops it and scores it |
| Video | A short clip with a visible face | Samples 32 frames, scores the face in each frame and averages the scores |
| Audio | A speech recording | Trims silence, takes up to six 2-second pieces, scores each and averages the scores |

Press **Analyse**. Results usually appear in under a second for images and audio and in a few seconds for video.

## 3. Read the result

- **Label:** *Real* (looks authentic) or *Fake / Deepfake* (likely AI-generated or manipulated).
- **Confidence ring:** how sure the model is about the label.
- **Fake score:** the average probability that the file is fake. Scores close to 50% are uncertain.
- **Video and audio charts:** one bar per frame or audio segment. Red bars are above the 50% fake threshold, green bars are below it. A file can have a "real" label and still contain a few red bars.

## 4. If you see an error

| Message | Likely reason | What to do |
|---|---|---|
| Unsupported file type | The format is not accepted | Convert the file or try another one |
| File does not match the tab | For example an audio file in the Image tab | Use the correct tab |
| No face can be seen | The face is too small, hidden or turned away (image and video) | Use a clearer file |
| File too large | The file is above the size limit | Use a shorter or smaller file |

## 5. Good to know

- The models are research prototypes. Results are **guidance, not proof**. Do not use them as evidence for important decisions.
- Test accuracy: image 90.9% (3,010 images), video 88.9% (18 videos, a small test), audio 83.0% (800 clips).
- Audio is weaker on fake voices from unfamiliar generators. Video was tested on only one face-swap method.
- Heavy compression, noise and poor lighting make results less reliable.
- Uploaded files are saved to a temporary file only while they are analysed, and are deleted straight after.

## 6. Trying it yourself

To test the app, use your own examples: a few real photos, videos and voice recordings, and a few AI-generated ones. Check how the label and the scores change. Dataset sources are listed in the project README.
