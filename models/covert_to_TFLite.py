import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import os

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255 , test_images / 255

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('cifar_model.h5')

train_images = tf.cast(train_images, tf.float32)
cifar_ds = tf.data.Dataset.from_tensor_slices((train_images)).batch(1)
def representative_dataset():
  for input_value in cifar_ds.take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.float32  # or tf.uint8
#converter.inference_output_type = tf.float32  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('cifar_cnn.tflite', 'wb') as f:
  f.write(tflite_model)

os.system("xxd -i cifar_cnn.tflite > cifar_cnn.tflite.cpp")

print("Done!")
