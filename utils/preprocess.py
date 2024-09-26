# utils/preprocess.py

import tensorflow as tf
import cv2

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load directly as grayscale
    
    # Resize the image to the desired shape
    resized_image = cv2.resize(image, (48, 48))
    
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image.astype('float32') / 255.0
    
    # Expand dimensions to add batch size and channel (required for grayscale images)
    input_image = normalized_image[np.newaxis, ..., np.newaxis]
    
    return tf.convert_to_tensor(input_image)  # Convert to TensorFlow tensor
