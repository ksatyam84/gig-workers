import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils
import time

# Constants for thresholds and duration requirements
EYE_AR_THRESH = 0.25              # Eyes are considered closed if EAR is below this
EYE_CLOSED_DURATION_THRESH = 2.0  # Seconds eyes must remain closed to be labeled "Sleepy"
MOUTH_AR_THRESH = 0.7             # Mouth Aspect Ratio threshold for yawning

# Variables to accumulate closed-eye duration
closed_duration = 0.0
last_time = time.time()

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/kumarsatyam/Desktop/face/shape_predictor_68_face_landmarks.dat")

# Define landmark indices for the eyes and inner mouth (68-point model)
(lStart, lEnd) = (42, 48)  # left eye (points 42-47)
(rStart, rEnd) = (36, 42)  # right eye (points 36-41)
(mStart, mEnd) = (60, 68)  # inner mouth (points 60-67)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between vertical landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the distance between horizontal landmarks
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute a simple ratio for yawning: vertical distance / horizontal distance
    A = dist.euclidean(mouth[2], mouth[6])  # Approximate vertical (points 62 and 66)
    C = dist.euclidean(mouth[0], mouth[4])  # Approximate horizontal (points 60 and 64)
    mar = A / C
    return mar

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate time elapsed between frames
    current_time = time.time()
    delta = current_time - last_time
    last_time = current_time

    # Resize frame and convert to grayscale for processing
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = detector(gray, 0)
    state = "Normal"
    
    for rect in rects:
        shape = predictor(gray, rect)
        # Convert the dlib shape to a NumPy array of (x, y) coordinates
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        
        # Extract eye coordinates and compute EAR for both eyes
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Extract mouth coordinates and compute MAR
        mouth = shape_np[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Accumulate closed-eye duration if EAR is below threshold
        if ear < EYE_AR_THRESH:
            closed_duration += delta
        else:
            closed_duration = 0.0  # Reset if eyes are open

        # Determine state based on accumulated closed time
        if closed_duration >= EYE_CLOSED_DURATION_THRESH:
            state = "Sleepy"
        elif ear < EYE_AR_THRESH:
            state = "Eyes Closed"
        else:
            state = "Awake"

        # Override state if yawning is detected (mouth open)
        if mar > MOUTH_AR_THRESH:
            state = "Yawning"

        # Draw the face rectangle and state text on the frame
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"State: {state}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Optionally, draw facial landmarks for debugging
        for (x_point, y_point) in shape_np:
            cv2.circle(frame, (x_point, y_point), 1, (0, 0, 255), -1)

    # Display the output frame
    cv2.imshow("Real-Time Sleepiness & Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
