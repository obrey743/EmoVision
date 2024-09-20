############################
#  *Project: EmoVison       #
#  * By: Obrey Muchena      #
#  * Date: 19 Sep 2024      #
############################

import cv2  # Import OpenCV library for computer vision tasks
from fer import FER  # Import FER (Facial Emotion Recognition) for detecting emotions

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Initialize the FER detector with MTCNN (Multi-task Cascaded Convolutional Neural Networks) for better face detection
detector = FER(mtcnn=True)

# Start an infinite loop to continuously capture frames from the webcam
while True:
    # Capture a single frame from the webcam
    ret, frame = cap.read()  # ret is a boolean indicating if the frame was captured successfully
    
    # If the frame was not captured, exit the loop
    if not ret:
        break
    
    # Use the FER detector to detect emotions in the current frame
    emotions = detector.detect_emotions(frame)
    
    # Loop through all detected faces in the frame
    for face in emotions:
        box = face['box']  # 'box' contains the coordinates of the face bounding box
        # Get the emotion with the highest score (most likely emotion)
        emotion, score = max(face['emotions'].items(), key=lambda x: x[1])
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        
        # Display the detected emotion and its score above the rectangle
        cv2.putText(frame, f"{emotion} ({score:.2f})", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Show the frame with the detected faces and emotions in a window titled 'EmoVision'
    cv2.imshow('EmoVision', frame)
    
    # Check if the 'q' key is pressed; if so, break the loop to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()