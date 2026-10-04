import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2  

folder = "images"

# images = []

# for file in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder, file))   
#     if img is not None:
#         images.append(img)

images = []

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (512, 512))   
    images.append(img)

images = np.array(images)

print("New Shape:", images.shape)
print("Dtype:", images.dtype)



# print(len(images))
# print(images[10].shape)
# print(images[100].shape)
# print(images[1000].shape)
# print(images[892].shape)
# print(images[1082].shape)
# print(images[0].dtype)

shapes = set()

for img in images:
    shapes.add(img.shape)

print(shapes)
# print(images)

# cv2.imshow("Sample Image", images[100])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


hsv_images = []

for img in images:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   #  BGR → HSV
    hsv_images.append(hsv)

hsv_images = np.array(hsv_images)

print("HSV Shape:", hsv_images.shape)
print("HSV dtype:", hsv_images.dtype)
# print(hsv_images)



# displaying all the HSV Channels separately 

# plt.figure(figsize=(12,4))

# plt.subplot(1,3,1)
# plt.imshow(hsv_images[0][:,:,0])
# plt.title("Hue")

# plt.subplot(1,3,2)
# plt.imshow(hsv_images[0][:,:,1])
# plt.title("Saturation")

# plt.subplot(1,3,3)
# plt.imshow(hsv_images[0][:,:,2])
# plt.title("Value")

# plt.show()


# h = hsv_images[0][:,:,0]
img = images[1790]
# normal BGR image
center = img[150:350, 150:350]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(center, cv2.COLOR_BGR2RGB))
plt.title("Center Region - Original Image")
plt.axis("off")
plt.show()
print("Mean Hue (center region):", np.mean(center))


# img = images[1790]
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower = np.array([0, 20, 50])
# upper = np.array([35, 255, 255])

# mask = cv2.inRange(hsv, lower, upper)

# plt.imshow(mask, cmap='gray')
# plt.title("Skin Mask")
# plt.show()