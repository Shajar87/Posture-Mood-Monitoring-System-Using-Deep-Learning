import cv2
import numpy as np
from keras.models import model_from_json

class Emotions:
    def __init__(self):
        # Load emotion detection model
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("emotiondetector.h5")
    
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.labels = {0: 'disgust', 1: 'happy', 2: 'neutral', 3: 'sad'}

    def detect_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no face is detected, return None
        if len(faces) == 0:
            return None, frame

        # Only consider the first face detected
        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI for the model
        try:
            face_roi = cv2.resize(face_roi, (48, 48))
        except Exception as e:
            print(str(e))
            return None, frame

        # Normalize the image
        face_roi = face_roi / 255.0

        # Reshape the image to match model input shape
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Predict emotion
        predicted_class = np.argmax(self.model.predict(face_roi), axis=-1)

        # Get the predicted emotion label
        predicted_label = self.labels[predicted_class[0]]

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return predicted_label, frame
