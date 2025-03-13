# utils/preprocess.py

import tensorflow as tf
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(48, 48)):
    """
    Loads an image, preprocesses it by resizing, normalizing, and formatting it 
    for model input, then converts it to a TensorFlow tensor.
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired image dimensions (height, width).
    
    Returns:
        tf.Tensor: Preprocessed image tensor with shape (1, target_size[0], target_size[1], 1).
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    
    # Resize the image
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to the range [0, 1]
    image_normalized = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Reshape to add batch and channel dimensions (batch_size=1, channels=1)
    image_processed = image_normalized[np.newaxis, ..., np.newaxis]
    
    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(image_processed, dtype=tf.float32)
