import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

custom_layers = {'BatchNormalization': tf.keras.layers.BatchNormalization}
model = tf.keras.models.load_model('model.h5',custom_layers)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# def image_preprocessing(path):
#     class_indices = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
#                      'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22,
#                      'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
#                      'Y': 34, 'Z': 35, 'del': 36, 'nothing': 37, 'space': 38}
#
#     class_indices = {value: key for key, value in class_indices.items()}
#     img = cv2.imread(path)
#     img = cv2.resize(img, (224, 224))
#
#     img = img / 225.
#
#     img = np.expand_dims(img, axis=0)
#     pre = model.predict(img)
#     pred_class = np.argmax(pre)
#
#     return class_indices[pred_class]
