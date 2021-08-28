# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images, test_images

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="cifar_cnn.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def invoke(X_test, name):
  input_data = tf.cast(np.array(X_test), tf.float32)
  input_data = np.expand_dims(input_data, axis=0)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)
  print(name)

# Call invoke on first 6 pictures of test data
for i in range(10):
    invoke(train_images[i], train_labels[i][0])




# load and prepare the image
def load_image(image):
	# load the image
	#img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(image)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example_h5_model():
    # load model
	model = load_model('cifar_model.h5')
	# load the image
	for i in range(10):
		img = load_image(train_images[i])
		# predict the class
		result = model.predict_classes(img)
		print(result[0])

# entry point, run the example
run_example_h5_model()

print("Done!")