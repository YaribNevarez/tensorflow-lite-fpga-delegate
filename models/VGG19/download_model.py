import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

model = VGG19(weights='imagenet')

model.summary()

tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True, show_layer_names=True, expand_nested=True)

model.save('vgg19.h5')