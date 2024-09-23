# main.py

from models.emotion_detector import EmotionDetector
from utils.preprocess import preprocess_image

def main():
    # Initialize the Emotion Detector with the correct path to the model file
    detector = EmotionDetector('models/emotion_model.keras')

    # Path to the test image (ensure you have an image in the assets folder)
    image_path = 'assets/test_image.jpg'  # Change the path to your actual image

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Predict the emotion
    emotion = detector.predict_emotion(preprocessed_image)

    # Print the detected emotion
    print(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    main()
