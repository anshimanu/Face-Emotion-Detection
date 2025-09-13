import cv2
import tkinter as tk
from tkinter import Label, Button
from keras.models import model_from_json
import numpy as np
from PIL import Image, ImageTk

# Load pre-trained model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to process the face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion detection function
def detect_emotions():
    global webcam, label
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)  # Fix mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        face = gray[q:q+s, p:p+r]
        cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
        try:
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(frame, prediction_label, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print("Error:", e)

    # Convert frame to ImageTk format for Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, detect_emotions)

# Start webcam
def start_detection():
    global webcam
    webcam = cv2.VideoCapture(0)
    detect_emotions()

# Stop detection and release resources
def stop_detection():
    global webcam
    if webcam.isOpened():
        webcam.release()
    root.destroy()

# Create GUI
root = tk.Tk()
root.title("Face Emotion Recognition")
root.geometry("800x600")

# Video feed label
label = Label(root)
label.pack()

# Buttons
start_button = Button(root, text="Start Detection", command=start_detection, bg="green", fg="white", font=("Arial", 12))
start_button.pack(pady=20)

stop_button = Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white", font=("Arial", 12))
stop_button.pack()

root.mainloop()
