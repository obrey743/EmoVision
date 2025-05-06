############################
#  * Project: EmoVision    #
#  * By: Obrey Muchena     #
#  * Updated: 5 May 2025   #
############################

import cv2
import time
import datetime
import csv
import os
from collections import deque
from fer import FER


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not access your webcam")
    return cap


def initialize_detector():
    try:
        return FER(mtcnn=True)
    except Exception as e:
        raise RuntimeError(f"Error initializing FER detector: {e}")


def get_emotion_color(emotion):
    emotion_colors = {
        "angry": (0, 0, 255),
        "disgust": (0, 128, 0),
        "fear": (128, 0, 128),
        "happy": (255, 255, 0),
        "sad": (0, 0, 128),
        "surprise": (0, 255, 255),
        "neutral": (255, 255, 255)
    }
    return emotion_colors.get(emotion, (255, 255, 255))


def save_screenshot(frame, label=""):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_capture_{label}_{timestamp}.png" if label else f"emotion_capture_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as '{filename}'")


def log_emotion_data(emotions):
    if not emotions:
        return

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "emotion_log.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Face#", "Emotion", "Score"])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx, face in enumerate(emotions):
            for emotion, score in face['emotions'].items():
                writer.writerow([timestamp, idx + 1, emotion, f"{score:.2f}"])


def draw_emotion_data(frame, emotions):
    for face in emotions:
        box = face['box']
        emotion, score = max(face['emotions'].items(), key=lambda x: x[1])
        color = get_emotion_color(emotion)

        # Draw bounding box
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display primary emotion
        text = f"{emotion} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Display emotion probabilities
        y_offset = y + h + 20
        for em, sc in face['emotions'].items():
            cv2.putText(frame, f"{em}: {sc:.2f}", (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_emotion_color(em), 1)
            y_offset += 20


def display_overlay(frame):
    overlay_text = "Press 'q' to quit | 's' to save | Auto-snap on 'happy'"
    cv2.putText(frame, overlay_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    try:
        cap = initialize_camera()
        detector = initialize_detector()

        frame_count = 0
        start_time = time.time()
        fps_queue = deque(maxlen=10)

        print("Press 'q' to quit, 's' to save a screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.resize(frame, (640, 480))
            emotions = detector.detect_emotions(frame)

            draw_emotion_data(frame, emotions)
            display_overlay(frame)

            # Face count display
            cv2.putText(frame, f"Faces: {len(emotions)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Log emotion data
            log_emotion_data(emotions)

            # Auto-screenshot if dominant emotion is 'happy'
            for face in emotions:
                dominant_emotion, confidence = max(face['emotions'].items(), key=lambda x: x[1])
                if dominant_emotion == 'happy' and confidence > 0.9:
                    save_screenshot(frame, "happy")

            # FPS calculation (moving average)
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            fps_queue.append(fps)
            avg_fps = sum(fps_queue) / len(fps_queue)
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('EmoVision', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_screenshot(frame)

    except Exception as e:
        print(str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()
