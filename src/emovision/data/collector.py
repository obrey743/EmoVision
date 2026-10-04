import os
import cv2
import time
import argparse
from ..config import EMOTIONS_DEFAULT, IMG_SIZE, FACE_MARGIN, DATA_DIR, IMAGE_EXTS
from ..utils.face import get_face_detector, detect_faces, largest_face, crop_face
from ..utils.preprocess import preprocess_image


def count_images(folder: str) -> int:
    return len([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])


def save_face(gray, box, out_dir: str) -> str:
    proc = preprocess_image(crop_face(gray, box, FACE_MARGIN), IMG_SIZE)
    filename = os.path.join(out_dir, f"{time.time_ns()}.png")
    cv2.imwrite(filename, (proc.squeeze() * 255).astype("uint8"))
    return filename


def main():
    parser = argparse.ArgumentParser(description="Collect face images per emotion via webcam.")
    parser.add_argument("--out", default=DATA_DIR, help="Output root directory")
    parser.add_argument("--classes", nargs="+", default=EMOTIONS_DEFAULT, help="Emotion class names")
    parser.add_argument("--per-class", type=int, default=150, help="Images per class")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between saves in auto-capture mode")
    args = parser.parse_args()

    for c in args.classes:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    detector = get_face_detector()

    print("Controls:")
    print("  space - capture one image")
    print("  a     - toggle auto-capture")
    print("  n / p - next / previous class")
    print("  q     - quit")
    print("=" * 40)

    class_idx = 0
    saved_counts = {c: count_images(os.path.join(args.out, c)) for c in args.classes}
    auto = False
    last_save = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(detector, gray)
            box = largest_face(faces)

            current_class = args.classes[class_idx]
            full = saved_counts[current_class] >= args.per_class

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                class_idx = (class_idx + 1) % len(args.classes)
                auto = False
                continue
            elif key == ord('p'):
                class_idx = (class_idx - 1) % len(args.classes)
                auto = False
                continue
            elif key == ord('a'):
                auto = not auto

            now = time.time()
            want_save = key == ord(' ') or (auto and now - last_save >= args.interval)
            if want_save and box is not None and not full:
                save_face(gray, box, os.path.join(args.out, current_class))
                saved_counts[current_class] += 1
                last_save = now
                if saved_counts[current_class] >= args.per_class:
                    auto = False
                    print(f"{current_class}: done ({args.per_class} images). Press 'n' for next class.")

            # Draw UI
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if (x, y, w, h) == box else (128, 128, 128)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            status = f"Class: {current_class}  [{saved_counts[current_class]}/{args.per_class}]"
            if full:
                status += "  FULL - press n"
            elif auto:
                status += "  AUTO"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if box is None:
                cv2.putText(frame, "No face detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("EmoVision Collector", frame)

            if all(saved_counts[c] >= args.per_class for c in args.classes):
                print("Collection complete.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Saved counts: " + ", ".join(f"{c}={n}" for c, n in saved_counts.items()))


if __name__ == "__main__":
    main()
