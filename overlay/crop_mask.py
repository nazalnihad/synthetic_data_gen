import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
mask_path = 'C:/Users/nazal/Downloads/pestvision_data/foreground_data/Detection_IP102/Masks/IP000000000.png'
img_path = 'C:/Users/nazal/Downloads/pestvision_data/foreground_data/Detection_IP102/JPEGImages/IP000000000.jpg'

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

# Threshold the mask to binary
_, mask_binary = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)

# Create an alpha channel from the mask
alpha = mask_binary

# Apply the mask to the original image
img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
img_rgba[:, :, 3] = alpha

# Crop the image to the bounding box of the non-zero region in the mask
y, x = np.where(alpha > 0)
top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
cropped_img = img_rgba[top:bottom+1, left:right+1]

# Convert BGRA to RGBA for displaying with matplotlib
cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGRA2RGBA)

# Display the cropped image
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(cropped_img_rgb)
plt.show()

# Save the cropped image
cv2.imwrite('cropped_image.png', cropped_img)