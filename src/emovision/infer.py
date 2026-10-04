import os
import json
import time
import argparse
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import IMG_SIZE, FACE_MARGIN, MODEL_PATH, LABELS_SUFFIX, EMOTIONS_DEFAULT
from .utils.face import Box, get_face_detector, detect_faces, crop_face
from .utils.preprocess import preprocess_image


def load_labels(model_path: str, num_classes: int) -> List[str]:
    path = model_path + LABELS_SUFFIX
    if os.path.exists(path):
        with open(path) as f:
            labels = json.load(f)
    else:
        print(f"Warning: {path} not found; falling back to default class names.")
        labels = EMOTIONS_DEFAULT
    if len(labels) != num_classes:
        raise SystemExit(f"Model has {num_classes} outputs but {len(labels)} labels were found.")
    return list(labels)


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceSmoother:
    """Exponential moving average of class probabilities per face, matched across frames by IoU."""

    def __init__(self, alpha: float = 0.4, min_iou: float = 0.3):
        self.alpha = alpha
        self.min_iou = min_iou
        self.tracks: Dict[int, tuple] = {}  # id -> (box, probs)
        self._next_id = 0

    def update(self, boxes: List[Box], probs: np.ndarray) -> List[np.ndarray]:
        new_tracks, out, used = {}, [], set()
        for box, p in zip(boxes, probs):
            best_id, best_iou = None, self.min_iou
            for tid, (tbox, _) in self.tracks.items():
                score = iou(box, tbox)
                if tid not in used and score >= best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id, smoothed = self._next_id, p
                self._next_id += 1
            else:
                smoothed = self.alpha * p + (1 - self.alpha) * self.tracks[best_id][1]
            used.add(best_id)
            new_tracks[best_id] = (box, smoothed)
            out.append(smoothed)
        self.tracks = new_tracks
        return out


def predict_faces(model, gray: np.ndarray, boxes: List[Box]) -> np.ndarray:
    if not boxes:
        return np.empty((0, model.output_shape[-1]), dtype=np.float32)
    batch = np.stack([preprocess_image(crop_face(gray, b, FACE_MARGIN), IMG_SIZE) for b in boxes])
    return model.predict_on_batch(batch)


def draw_predictions(frame, boxes: List[Box], probs: List[np.ndarray], labels: List[str],
                     min_conf: float, show_bars: bool = True) -> None:
    for (x, y, w, h), p in zip(boxes, probs):
        idx = int(np.argmax(p))
        conf = float(p[idx])
        text = f"{labels[idx]} {conf:.0%}" if conf >= min_conf else "Uncertain"
        color = (0, 255, 0) if conf >= min_conf else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if show_bars:
            bar_x = x + w + 10
            for i, (name, v) in enumerate(zip(labels, p)):
                by = y + i * 18
                cv2.rectangle(frame, (bar_x, by), (bar_x + int(100 * v), by + 12), (255, 160, 0), -1)
                cv2.putText(frame, f"{name} {v:.2f}", (bar_x + 105, by + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def run_image(model, labels, detector, path: str, min_conf: float, out: Optional[str]) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Failed to read image: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = [tuple(map(int, b)) for b in detect_faces(detector, gray, min_size=32)]
    probs = predict_faces(model, gray, boxes)
    if not boxes:
        print("No faces detected.")
    for i, (box, p) in enumerate(zip(boxes, probs)):
        ranked = sorted(zip(labels, p), key=lambda t: -t[1])
        print(f"Face {i} at {box}: " + ", ".join(f"{n}={v:.2f}" for n, v in ranked))
    draw_predictions(frame, boxes, list(probs), labels, min_conf)
    if out:
        cv2.imwrite(out, frame)
        print(f"Saved annotated image to {out}")
    else:
        cv2.imshow("EmoVision", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(model, labels, detector, camera: int, min_conf: float, alpha: float) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    smoother = FaceSmoother(alpha=alpha)
    show_bars = True
    fps, last = 0.0, time.time()
    print("Controls:  b - toggle probability bars   q - quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = [tuple(map(int, b)) for b in detect_faces(detector, gray)]
            probs = smoother.update(boxes, predict_faces(model, gray, boxes))
            draw_predictions(frame, boxes, probs, labels, min_conf, show_bars)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(frame, f"FPS {fps:.1f}  Faces {len(boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("EmoVision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                show_bars = not show_bars
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run emotion recognition on a webcam or an image.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--image", help="Run on a single image instead of the webcam")
    parser.add_argument("--out", help="With --image: save the annotated result here instead of showing it")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--min-conf", type=float, default=0.4, help="Below this confidence, show 'Uncertain'")
    parser.add_argument("--smooth", type=float, default=0.4,
                        help="EMA weight for new predictions (1.0 = no smoothing)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train one first with: python main.py train")

    from tensorflow import keras
    model = keras.models.load_model(args.model)
    labels = load_labels(args.model, model.output_shape[-1])
    detector = get_face_detector()

    if args.image:
        run_image(model, labels, detector, args.image, args.min_conf, args.out)
    else:
        run_camera(model, labels, detector, args.camera, args.min_conf, args.smooth)


if __name__ == "__main__":
    main()
