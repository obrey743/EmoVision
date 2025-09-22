import os
import cv2
import time
import argparse
from ..config import EMOTIONS_DEFAULT, IMG_SIZE, CASCADE_PATH
from ..utils.preprocess import preprocess_image

def get_face_detector():
    if CASCADE_PATH and os.path.exists(CASCADE_PATH):
        return cv2.CascadeClassifier(CASCADE_PATH)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade)

def main():
    parser = argparse.ArgumentParser(description="Collect face images per emotion via webcam.")
    parser.add_argument("--out", default="data/dataset", help="Output root directory")
    parser.add_argument("--classes", nargs="+", default=EMOTIONS_DEFAULT, help="Emotion class names")
    parser.add_argument("--per-class", type=int, default=150, help="Images per class")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for c in args.classes:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    detector = get_face_detector()

    print("Controls:")
    print("  n - next class")
    print("  q - quit")
    print("  space - capture")
    print("="*40)

    class_idx = 0
    saved_counts = {c: len([f for f in os.listdir(os.path.join(args.out, c)) if f.lower().endswith(('.png','.jpg','.jpeg'))]) for c in args.classes}
    last_save = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(64,64))

        current_class = args.classes[class_idx]
        status = f"Class: {current_class}  [{saved_counts[current_class]}/{args.per_class}]"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Collector - press 'space' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            class_idx = (class_idx + 1) % len(args.classes)
        elif key == ord(' '):
            if len(faces) == 0:
                continue
            (x,y,w,h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
            face_gray = gray[y:y+h, x:x+w]
            proc = preprocess_image(face_gray, IMG_SIZE)

            cls = current_class
            count = saved_counts[cls]
            if count < args.per_class:
                out_dir = os.path.join(args.out, cls)
                filename = os.path.join(out_dir, f"{int(time.time()*1000)}.png")
                to_save = (proc.squeeze() * 255).astype("uint8")
                cv2.imwrite(filename, to_save)
                saved_counts[cls] += 1

        if all(saved_counts[c] >= args.per_class for c in args.classes):
            print("Collection complete.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
