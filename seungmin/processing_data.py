import numpy as np
import os
import cv2

print(os.listdir("/Users/hongseongmin/Desktop/data_image/"))

img = cv2.imread("/Users/hongseongmin/Desktop/data_image/test0.png")
print(img.shape)