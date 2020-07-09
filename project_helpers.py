import keras
import tensorflow
import numpy as np
import cv2
from keras.models import load_model
from time import sleep 
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import pathlib 
from pathlib import Path


import os
import requests
import tensorflow as tf
from flask import Flask, request, jsonify

root = Path(".")
path_to_model = root / "modelos/Emotion_little_vgg.h5"
path_to_cascade = root / "modelos/haarcascade_frontalface_defaults.xml"

face_detector = cv2.CascadeClassifier(str(path_to_cascade))
classifier = load_model(path_to_model)

emotion_labels = ("Angry", "Happy", "Neutral", "Sad", "Surprise")

print(emotion_labels)


#Paso 3: definiendo aplicacion de flask
app = Flask(__name__)

#Funcion de clasificacion para API
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def emotion_detect(img_name):
	
    image = cv2.imread(img_name)
    emotions = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(image, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi]) != 0:

            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predict = classifier.predict(roi)[0]
            predict = predict.argmax()
            emotion = emotion_labels[predict]
            emotions.append(emotion)

        else:
            continue

    return jsonify({"emocion_detectada": emotions})


# Iniciar App de flask y hacer predicciones
app.run(port=5000, debug=False)










































