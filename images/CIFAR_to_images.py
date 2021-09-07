from tensorflow.keras import datasets
import os

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

test_labels.tofile("labels")

# convert from integers to floats
test_norm = test_images.astype('float32')
# normalize to range 0-1
test_norm = test_norm / 255.0

if not os.path.exists('CIFAR'):
  os.mkdir("CIFAR")

def save_binary(image, name):
  filename = "CIFAR/" + name
  image.tofile(filename)

for i in range(len(test_norm)):
  save_binary(test_norm[i], str(i))

print("Done!")
