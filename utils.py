# utils.py

import cv2
import numpy as np

def preprocess_image(image, target_size=(48, 48)):
    """Resize and normalize the image for emotion detection."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=[0, -1])
    return image

def draw_emotion_label(image, text, x, y):
    """Draw the emotion label on the image."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
