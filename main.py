# main.py

from models.emotion_detector import EmotionDetector
from utils.preprocess import preprocess_image

def load_emotion_detector(model_path):
    """Initialize the Emotion Detector with the given model path."""
    return EmotionDetector(model_path)

def predict_emotion_from_image(detector, image_path):
    """Preprocess the image and predict the emotion using the detector."""
    preprocessed_image = preprocess_image(image_path)
    return detector.predict_emotion(preprocessed_image)

def main():
    model_path = 'models/emotion_model.keras'  # Path to the model file
    image_path = 'assets/test_image.jpg'  # Path to the test image

    # Initialize the Emotion Detector
    detector = load_emotion_detector(model_path)

    # Predict the emotion
    emotion = predict_emotion_from_image(detector, image_path)

    # Print the detected emotion
    print(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    main()
