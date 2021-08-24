import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np

model = InceptionV3(weights='imagenet')

model.summary()

model.save('inception.h5')

tf.keras.utils.plot_model(model, "inception.png", show_shapes=True, show_layer_names=True, expand_nested=True)
