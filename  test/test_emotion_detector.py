import unittest
from models.emotion_detector import EmotionDetector
import numpy as np

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        # Initialize EmotionDetector with a valid model path
        self.detector = EmotionDetector('models/emotion_model.keras')

    def test_predict_emotion(self):
        # Create a dummy 48x48 grayscale image
        dummy_image = np.random.rand(48, 48, 1).astype('float32')
        dummy_image = np.expand_dims(dummy_image, axis=0)
        
        # Run prediction
        emotion = self.detector.predict_emotion(dummy_image)
        
        # Check if output is a string representing an emotion
        self.assertIsInstance(emotion, str)
        self.assertIn(emotion, ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral'])

if __name__ == '__main__':
    unittest.main()
