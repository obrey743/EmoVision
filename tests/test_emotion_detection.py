# tests/test_emotion_detection.py

import unittest
from models.emotion_detector import EmotionDetector
import numpy as np

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EmotionDetector()
        self.sample_image = np.random.rand(1, 48, 48, 1)

    def test_prediction(self):
        result = self.detector.predict(self.sample_image)
        self.assertIn(result, self.detector.emotions)

if __name__ == "__main__":
    unittest.main()
