import cv2
import numpy as np

def load_and_preprocess(path: str, size: int = 48) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return preprocess_image(img, size)

def preprocess_image(gray_img: np.ndarray, size: int = 48) -> np.ndarray:
    # Resize, scale [0,1], add channel dimension
    img = cv2.resize(gray_img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # H W 1
    return img
