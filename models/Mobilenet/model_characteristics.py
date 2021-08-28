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
interpreter = tf.lite.Interpreter(model_path="mobilenetv2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
layer_details = interpreter.get_tensor_details()
for layer in layer_details:
    name = layer['name']
    if name.find("Conv2D") != -1:
      print("\nLayer Name: {}".format(layer['name']))
      print("\tIndex: {}".format(layer['index']))
      print("\n\tShape: {}".format(layer['shape']))
      print("\tTensor: {}".format(interpreter.get_tensor(layer['index']).shape))
      print("\tTensor Type: {}".format(interpreter.get_tensor(layer['index']).dtype))
      print("\tQuantisation Parameters")
      print("\t\tScales: {}".format(layer['quantization_parameters']['scales'].shape))
      print("\t\tScales Type: {}".format(layer['quantization_parameters']['scales'].dtype))
      print("\t\tZero Points: {}".format(layer['quantization_parameters']['zero_points']))
      print("\t\tQuantized Dimension: {}".format(layer['quantization_parameters']['quantized_dimension']))


print("Done!")