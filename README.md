# gig-workers

# Face Detection and Drowsiness/Yawn Recognition

This module uses computer vision techniques to detect faces and monitor eye and mouth activity in real time. It aims to determine the user's state (e.g., awake, eyes closed, sleepy, yawning) using facial landmarks with the following features:

- **Real-Time Detection:** Captures webcam feed to process each video frame.
- **Drowsiness Detection:** Calculates the Eye Aspect Ratio (EAR) to detect if the eyes are closed long enough to indicate sleepiness.
- **Yawn Detection:** Computes the Mouth Aspect Ratio (MAR) to detect yawning.
- **Visualization:** Draws bounding boxes around detected faces and overlays the current state.

## How It Works

1. The script loads dlib's pre-trained facial landmark predictor ([`predictor`](face/face.py)) and a face detector.
2. It computes the EAR by measuring vertical and horizontal distances between eye landmarks.
3. It measures the MAR by comparing vertical and horizontal distances across mouth landmarks.
4. Based on computed thresholds, it determines if a person is "Awake", "Eyes Closed", "Sleepy", or "Yawning".

## Prerequisites

- Python 3.x
- OpenCV
- dlib
- numpy
- scipy
- imutils

Make sure to install the required packages, for example:

```

# Gorkapi Food Analysis API

This Flask application provides an endpoint to analyze food images for nutritional insights. It uses the Ollama API (with the "llava:7b" model) to determine if the food is healthy, reporting key nutritional information, and it converts the resulting text response into an audio file.

## Features

- **Image Upload:** Accepts a food image via a POST request at the `/analyze_food` endpoint.
- **Nutritional Analysis:** Uses the Ollama API to analyze the image and provide information on calories and nutrients.
- **Audio Feedback:** Converts the textual analysis into an audio file using Google Text-to-Speech (gTTS).

## Prerequisites

- Python 3.x
- Flask
- playsound
- gTTS

Install the required packages with:

```

see the demo video for more info

