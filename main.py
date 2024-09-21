# main.py

import cv2
from utils import preprocess_image, draw_emotion_label
from models.emotion_detector import EmotionDetector

# Load the face detection model

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
detector = EmotionDetector()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        processed_face = preprocess_image(face)
        emotion = detector.predict(processed_face)
        draw_emotion_label(frame, emotion, x, y - 10)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("EmoVision", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
