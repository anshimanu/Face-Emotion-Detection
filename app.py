from flask import Flask, render_template, Response
import cv2
import threading
import atexit
import webbrowser
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

with open("emotiondetector.json","r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

labels = {0: 'angry', 1: 'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, im = camera.read()
        if not success:
            break
        else:
            im = cv2.flip(im, 1)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (p, q, r, s) in faces:
                face = gray[q: q+s, p: p+r]
                cv2.rectangle(im, (p,q), (p+q, q+s), (255, 0, 0), 2)
                try:
                    face = cv2.resize(face, (48, 48))
                    img = extract_features(face)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception as e:
                    print("Error processing face:", e)
            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')            
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

def cleanup():
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()
    print("Camera released and resources cleaned up.")

atexit.register(cleanup)

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)

# import atexit

# @atexit.register
# def cleanup():
#     camera.release()
#     cv2.destroyAllWindows()