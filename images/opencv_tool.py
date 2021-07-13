import numpy as np
import argparse
import cv2
import os


name = "dog.jpg"
img_ori = cv2.imread(name)
img_t = cv2.resize(img_ori, tuple([96, 96]))
# Three channels
#img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
# Single channel
img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
img = np.asarray(img_t, np.int8)

img.astype('int8').tofile(name + ".bin")

os.system("xxd -i " + name + ".bin > " + name + ".cpp")

cv2.imshow(name, img_t)
cv2.imwrite("resized" + name, img_t)
cv2.waitKey(0)

