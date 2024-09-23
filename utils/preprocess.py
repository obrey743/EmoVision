# utils/preprocess.py

import tensorflow as tf
import cv2

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Resize and convert to grayscale (adjust size based on your model's input requirement)
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to match model input (batch size, height, width, channels)
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension (grayscale)
    image = tf.expand_dims(image, axis=0)   # Add batch dimension
    return image

