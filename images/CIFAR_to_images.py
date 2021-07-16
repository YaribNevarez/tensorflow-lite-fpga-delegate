import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images , test_images 

def save_binary(name, image, label):
  filename = "CIFAR/" + name
  image.tofile(name)
  os.system("xxd -i " + name + " > " + filename)
  os.system("rm " + name)
  plt.imshow(image)
  plt.show()

for i in range(5):
  save_binary(str(i), test_images[i], test_labels[i][0])


print("Done!")
