# Deepfake Detection System

An AI system for detecting deepfake and AI-generated images, built as part of the 
UC ITS Capstone project. The current MVP focuses on **image-based deepfake detection**, 
using a fine-tuned pre-trained CNN to classify real vs. AI-generated/manipulated images, 
with an Explainable AI (XAI) component to visualise model decisions.

## Current Scope (MVP)

### Image Detection
Identifies AI-generated or manipulated images using a fine-tuned convolutional neural 
network, with Grad-CAM visualisation to explain predictions.

## Future Work (Stretch Goals)
- Video deepfake detection (facial expressions, lip sync, temporal inconsistencies)
- Audio/voice deepfake detection (spectrogram and MFCC-based analysis)

## Goals
- Detect previously unseen (novel) deepfake generation techniques
- Maintain accuracy under real-world conditions (compression, noise, lighting variation)
