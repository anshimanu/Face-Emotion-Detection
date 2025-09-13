import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function to prepare image for prediction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop for real-time emotion detection
while True:
    ret, im = webcam.read()
    im = cv2.flip(im, 1)  # Fix mirror image
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (p, q, r, s) in faces:
        face = gray[q:q+s, p:p+r]  # Extract the face region
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw rectangle around the face

        try:
            # Resize and preprocess the face image
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)

            # Predict emotion
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Display emotion label on the frame
            cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print("Error processing face:", e)

    # Display the video feed with detected emotions
    cv2.imshow("Output", im)

    # Exit the loop if 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        print("Terminating the GUI...")
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
