import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# for keras
from classification_models.tfkeras import Classifiers

model_names = Classifiers.models_names()

print(model_names)

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18((224, 224, 3), weights='imagenet')

model.summary()

model.save('resnet18.h5')

tf.keras.utils.plot_model(model, "resnet18.png", show_shapes=True, show_layer_names=True, expand_nested=True)
