import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

model = keras.models.load_model('alexnet_xray')

img = keras.preprocessing.image.img_to_array(load_img("chest_xray/val/BACTERIA/person1946_bacteria_4875.jpeg", target_size=(227, 227)))

img = np.array([img])/255

prob = model.predict(img)[0][0]

prob = round(prob*100, 2)

if prob > 50:

    print(str(prob)+"%"+" PNEUMONIA")

elif prob<50:

    print(str(100-prob) + "%" + " NORMAL")

else:

    print("UNCERTAIN")


