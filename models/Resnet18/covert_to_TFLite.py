import tensorflow as tf

from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import os

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('resnet18.h5')

model.summary()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.float32  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('resnet18.tflite', 'wb') as f:
  f.write(tflite_model)

os.system("xxd -i resnet18.tflite > resnet18.tflite.cpp")

print("Done!")
