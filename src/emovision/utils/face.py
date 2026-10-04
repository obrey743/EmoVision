import os
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import CASCADE_PATH

Box = Tuple[int, int, int, int]


def get_face_detector(cascade_path: Optional[str] = CASCADE_PATH) -> cv2.CascadeClassifier:
    if cascade_path and os.path.exists(cascade_path):
        path = cascade_path
    else:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"Failed to load face cascade: {path}")
    return detector


def detect_faces(detector: cv2.CascadeClassifier, gray: np.ndarray, min_size: int = 64) -> np.ndarray:
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(min_size, min_size))
    return np.asarray(faces, dtype=int).reshape(-1, 4)


def largest_face(faces: np.ndarray) -> Optional[Box]:
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def crop_face(gray: np.ndarray, box: Box, margin: float = 0.1) -> np.ndarray:
    """Crop a face with a small margin, clamped to the image bounds."""
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    H, W = gray.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return gray[y0:y1, x0:x1]
