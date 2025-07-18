import cv2
from keras.models import load_model
from utils import decode_emotion, preprocess_roi
import serial

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.h5")
arduino = serial.Serial('COM3', 9600)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_processed = preprocess_roi(roi)
        prediction = emotion_model.predict(roi_processed)
        mood = decode_emotion(prediction)

        if mood == "Happy":
            arduino.write(b'L')  # Light
        elif mood == "Sad":
            arduino.write(b'W')  # Water
        elif mood == "Angry":
            arduino.write(b'S')  # Shade

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
