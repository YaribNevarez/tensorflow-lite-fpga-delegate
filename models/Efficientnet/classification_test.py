# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import matplotlib.pyplot as plt
import numpy as np


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="efficientnet_lite0_fp32_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

x.tofile('elephant.dat')

interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('TensorFlow Lite Predicted:', decode_predictions(output_data, top=3)[0])

print("Done!")