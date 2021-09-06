import tensorflow as tf

from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import os

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('sconv.h5')

model.summary()

tf.keras.utils.plot_model(model, "sconv.png", show_shapes=True, show_layer_names=True, expand_nested=True)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255 , test_images / 255

train_images = tf.cast(train_images, tf.float32)
cifar_ds = tf.data.Dataset.from_tensor_slices((train_images)).batch(1)
def representative_dataset():
  for input_value in cifar_ds.take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # or tf.uint8
converter.inference_output_type = tf.float32  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('sconvi8.tflite', 'wb') as f:
  f.write(tflite_model)

#os.system("xxd -i sconv.tflite > sconv.tflite.cpp")

print("Done!")
