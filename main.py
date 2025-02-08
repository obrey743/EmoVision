############################
#  * Project: EmoVision    #
#  * By: Obrey Muchena     #
#  * Date: 19 Sep 2024     #
############################

import cv2  # OpenCV for image processing
import time  # To calculate FPS
from fer import FER  # Facial Emotion Recognition

# Initialize the webcam (0 is default)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

# Initialize FER detector with MTCNN for better face detection
detector = FER(mtcnn=True)

# Emotion-to-color mapping
emotion_colors = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 128, 0),    # Green
    "fear": (128, 0, 128),     # Purple
    "happy": (255, 255, 0),    # Yellow
    "sad": (0, 0, 128),        # Dark Blue
    "surprise": (0, 255, 255), # Cyan
    "neutral": (255, 255, 255) # White
}

# Frame count for FPS calculation
frame_count = 0
start_time = time.time()

# Start video capture loop
while True:
    ret, frame = cap.read()  # Capture frame
    
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    # Loop through detected faces
    for face in emotions:
        box = face['box']  # Face bounding box
        emotion, score = max(face['emotions'].items(), key=lambda x: x[1])  # Most likely emotion
        color = emotion_colors.get(emotion, (255, 255, 255))  # Default to white

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)

        # Display main emotion with background for better visibility
        text = f"{emotion} ({score:.2f})"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (box[0], box[1] - h - 10), (box[0] + w, box[1]), color, -1)  # Background
        cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Display all emotion probabilities below the box
        y_offset = box[1] + box[3] + 20
        for em, sc in face['emotions'].items():
            text = f"{em}: {sc:.2f}"
            cv2.putText(frame, text, (box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_colors.get(em, (255, 255, 255)), 1)
            y_offset += 20

    # FPS Calculation
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show output
    cv2.imshow('EmoVision', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save screenshot
        cv2.imwrite("emotion_capture.png", frame)
        print("Screenshot saved as 'emotion_capture.png'")

# Release resources
cap.release()
cv2.destroyAllWindows()
