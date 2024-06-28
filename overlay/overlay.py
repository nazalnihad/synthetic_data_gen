import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# Load images
s_img = cv2.imread("D:/projects/img_blend/cropped_image.png", cv2.IMREAD_UNCHANGED)
l_img = cv2.imread("C:/Users/nazal/Downloads/pestvision_data/background_data/paddy_disease_classification/val/normal/24700.jpg")

s_height, s_width = s_img.shape[:2]
l_height, l_width = l_img.shape[:2]

# Convert the background image to grayscale
gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection to find edges
edges = cv2.Canny(gray, 100, 200)

# Find coordinates of edge pixels
edge_coords = np.column_stack(np.where(edges > 0))

# Select a random edge coordinate to position the small image
random_edge = edge_coords[random.randint(0, len(edge_coords) - 1)]
y_offset, x_offset = random_edge

# Margin from edge
margin = 50

# Calculate the maximum size for the overlay image (1/8th of the background image)
max_overlay_size = (l_width // 8, l_height // 8)

# Ensure the small image fits within the background dimensions and does not exceed margins
max_scale_width = max_overlay_size[0] / s_width
max_scale_height = max_overlay_size[1] / s_height
max_scale = min(max_scale_width, max_scale_height)

# Random scaling factor (0.5 to max_scale)
scale = random.uniform(0.5, max_scale)

# Resize the small image
new_width = int(s_width * scale)
new_height = int(s_height * scale)
s_img_resized = cv2.resize(s_img, (new_width, new_height))

# Ensure the overlay fits within the margins
if y_offset < margin:
    y_offset = margin
if y_offset + new_height > l_height - margin:
    y_offset = l_height - new_height - margin

if x_offset < margin:
    x_offset = margin
if x_offset + new_width > l_width - margin:
    x_offset = l_width - new_width - margin

# Overlay the small image onto the background image
if s_img_resized.shape[2] == 4:
    # Separate the alpha channel
    alpha_s = s_img_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Blend the images
    for c in range(0, 3):
        l_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = (
            alpha_s * s_img_resized[:, :, c] + 
            alpha_l * l_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c]
        )
else:
    # If there's no alpha channel, just overlay the image
    l_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = s_img_resized

# Convert BGR to RGB for displaying with matplotlib
l_img_rgb = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)

# Display the images
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Edge Detection')
plt.axis('off')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Overlayed Image')
plt.axis('off')
plt.imshow(l_img_rgb)

plt.show()

print(f"Placed at position ({x_offset}, {y_offset}) with scale {scale:.2f}")
