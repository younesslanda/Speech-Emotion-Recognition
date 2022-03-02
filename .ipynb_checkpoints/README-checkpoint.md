# Speech-Emotion-Recognition

## Introduction
This repo contains the code for speech preprocessing and feature extraction for speech emotion detection using `torchaudio`, which is a library for audio and signal processing with PyTorch. It provides I/O, signal and data processing functions, datasets, model implementations and application components.. </br>

## Dataset
The dataset that is used is `RAVDESS` dataset (The Ryerson Audio-Visual Database of Emotional Speech and Song), that can be downloaded free of charge at  [this link](https://zenodo.org/record/1188976).</br>

The dataset have 7356 files (total size: 24.8 GB) and contains   24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  We only considered the Audio-only files.

## Model
The model 