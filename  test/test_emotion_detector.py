import unittest
import numpy as np
from models.emotion_detector import EmotionDetector

class EmotionDetectorTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the EmotionDetector model
        cls.detector = EmotionDetector('models/emotion_model.keras')

    def test_emotion_prediction(self):
        # Generate a random 48x48 grayscale image
        test_image = np.random.rand(48, 48, 1).astype(np.float32)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Predict emotion
        predicted_emotion = self.detector.predict_emotion(test_image)
        
        # Validate the prediction output
        self.assertIsInstance(predicted_emotion, str)
        self.assertIn(predicted_emotion, ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral'])

if __name__ == '__main__':
    unittest.main()
