# models/emotion_detector.py

import tensorflow as tf
from config import MODEL_PATH

class EmotionDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, image):
        prediction = self.model.predict(image)
        max_index = prediction.argmax()
        return self.emotions[max_index]
