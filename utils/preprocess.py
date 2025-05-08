# utils/preprocess.py

import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (48, 48)) -> tf.Tensor:
    """
    Loads and preprocesses a grayscale image for model input.

    Steps:
    - Load image in grayscale
    - Resize to target size
    - Normalize pixel values to [0, 1]
    - Reshape to (1, height, width, 1)
    - Convert to TensorFlow tensor

    Args:
        image_path (str): Path to the input image file.
        target_size (Tuple[int, int]): Desired dimensions (height, width).

    Returns:
        tf.Tensor: Preprocessed image tensor with shape (1, height, width, 1).
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # Ensure image is resized to target size
    try:
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        raise RuntimeError(f"Error resizing image: {e}")

    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Add batch and channel dimensions
    image_reshaped = image_normalized[np.newaxis, ..., np.newaxis]

    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(image_reshaped, dtype=tf.float32)
