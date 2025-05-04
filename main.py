############################
#  * Project: EmoVision    #
#  * By: Obrey Muchena     #
#  * Date: 19 Sep 2024     #
############################

import cv2
import time
import datetime
from fer import FER


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not access webcam")
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


def save_screenshot(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_capture_{timestamp}.png"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as '{filename}'")


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

        # Display all emotion scores
        y_offset = y + h + 20
        for em, sc in face['emotions'].items():
            cv2.putText(frame, f"{em}: {sc:.2f}", (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_emotion_color(em), 1)
            y_offset += 20


def main():
    try:
        cap = initialize_camera()
        detector = initialize_detector()

        frame_count = 0
        start_time = time.time()

        print("Press 'q' to quit, 's' to save a screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame = cv2.resize(frame, (640, 480))
            emotions = detector.detect_emotions(frame)

            draw_emotion_data(frame, emotions)

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
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
