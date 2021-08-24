import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

model = VGG16(weights='imagenet')

model.summary()

model.save('vgg16.h5')

tf.keras.utils.plot_model(model, "vgg16.png", show_shapes=True, show_layer_names=True, expand_nested=True)
