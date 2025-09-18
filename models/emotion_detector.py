# models/emotion_detector.py

import logging
from typing import List
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    A class for detecting emotions from preprocessed images using a trained TensorFlow model.
    """

    def __init__(self, model_path: str = "models/emotion_model.keras", 
                 classes: List[str] = None):
        """
        Initialize the EmotionDetector.

        Args:
            model_path (str): Path to the trained Keras model.
            classes (List[str], optional): Custom list of emotion labels.
        """
        self.classes = classes or ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral']

        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info("✅ Emotion model loaded successfully from %s", model_path)
        except Exception as e:
            logger.error("❌ Failed to load model from %s: %s", model_path, e)
            raise

    def predict_emotion(self, preprocessed_image: np.ndarray) -> str:
        """
        Predict the emotion for a given preprocessed image.

        Args:
            preprocessed_image (np.ndarray): Preprocessed input image, shaped for the model.

        Returns:
            str: Predicted emotion label.
        """
        try:
            prediction = self.model.predict(preprocessed_image, verbose=0)
            return self._decode_prediction(prediction)
        except Exception as e:
            logger.error("❌ Prediction failed: %s", e)
            raise

    def _decode_prediction(self, prediction: np.ndarray) -> str:
        """
        Convert model output into a human-readable emotion label.

        Args:
            prediction (np.ndarray): Model output probabilities.

        Returns:
            str: Predicted emotion label.
        """
        if prediction.ndim != 2 or prediction.shape[1] != len(self.classes):
            raise ValueError("Prediction shape does not match expected number of classes.")

        predicted_index = int(tf.argmax(prediction, axis=1).numpy()[0])
        predicted_class = self.classes[predicted_index]

        logger.info("🔎 Prediction probabilities: %s", prediction.tolist())
        logger.info("🎭 Predicted emotion: %s", predicted_class)

        return predicted_class
