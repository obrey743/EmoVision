# models/emotion_detector.py

import tensorflow as tf

class EmotionDetector:
    def __init__(self, model_path='models/emotion_model.keras'):
        try:
            # Load the pre-trained model from the provided path
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_emotion(self, preprocessed_image):
        # Predict emotion based on the preprocessed image
        prediction = self.model.predict(preprocessed_image)
        return self.decode_prediction(prediction)

    def decode_prediction(self, prediction):
        # Emotion classes based on the model's training
        classes = ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral']
        predicted_class = classes[tf.argmax(prediction, axis=1).numpy()[0]]
        return predicted_class
