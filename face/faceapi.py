from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils
import time

app = Flask(__name__)

# Constants for thresholds and duration requirements
EYE_AR_THRESH = 0.25
EYE_CLOSED_DURATION_THRESH = 2.0
MOUTH_AR_THRESH = 0.7

# Variables to accumulate closed-eye duration
closed_duration = 0.0
last_time = time.time()

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/kumarsatyam/Desktop/face/shape_predictor_68_face_landmarks.dat")

# Define landmark indices for the eyes and inner mouth (68-point model)
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
(mStart, mEnd) = (60, 68)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[0], mouth[4])
    mar = A / C
    return mar

@app.route('/detect', methods=['POST'])
def detect():
    global closed_duration, last_time

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    current_time = time.time()
    delta = current_time - last_time
    last_time = current_time

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    state = "Normal"
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mouth = shape_np[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            closed_duration += delta
        else:
            closed_duration = 0.0

        if closed_duration >= EYE_CLOSED_DURATION_THRESH:
            state = "Sleepy"
        elif ear < EYE_AR_THRESH:
            state = "Eyes Closed"
        else:
            state = "Awake"

        if mar > MOUTH_AR_THRESH:
            state = "Yawning"

    return jsonify({"state": state})

if __name__ == '__main__':
    app.run(debug=True,port=8080)