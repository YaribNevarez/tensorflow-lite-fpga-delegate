import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np

model = ResNet50(weights='imagenet')

model.summary()

tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True, show_layer_names=True, expand_nested=True)

model.save('resnet50_imagenet.h5')