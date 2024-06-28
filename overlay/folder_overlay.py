import cv2
import numpy as np
import random
import os

def overlay_image(background, foreground, x_offset, y_offset):
    if foreground.shape[2] == 4:
        # Separate the alpha channel
        alpha_s = foreground[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # Blend the images
        for c in range(0, 3):
            background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1], c] = (
                alpha_s * foreground[:, :, c] + 
                alpha_l * background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1], c]
            )
    else:
        # If there's no alpha channel, just overlay the image
        background[y_offset:y_offset+foreground.shape[0], x_offset:x_offset+foreground.shape[1]] = foreground

    return background

def convert_to_yolo_format(x, y, width, height, img_width, img_height):
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height
    return (x_center, y_center, width, height)

def process_image(background_path, foreground_images, output_dir, labels_dir):
    # Read background image
    background = cv2.imread(background_path)
    
    # Get background dimensions
    bg_height, bg_width = background.shape[:2]
    
    # Calculate max size (1/7th of background)
    max_overlay_width = bg_width // 9
    max_overlay_height = bg_height // 9
    
    # Randomly select 1-5 foreground images
    num_overlays = random.randint(1, 5)
    selected_foregrounds = random.sample(foreground_images, num_overlays)
    
    annotations = []
    
    for fg_path in selected_foregrounds:
        # Read foreground image
        foreground = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        
        # Calculate scale to fit within 1/7th of background
        fg_height, fg_width = foreground.shape[:2]
        scale_width = max_overlay_width / fg_width
        scale_height = max_overlay_height / fg_height
        scale = min(scale_width, scale_height)
        
        # Random scaling factor (0.5 to calculated scale, not exceeding 1.0)
        min_scale = min(0.5, scale)
        max_scale = min(1.0, scale)
        random_scale = random.uniform(min_scale, max_scale)
        
        # Resize the foreground image
        new_width = int(fg_width * random_scale)
        new_height = int(fg_height * random_scale)
        foreground_resized = cv2.resize(foreground, (new_width, new_height))
        
        # Random position (with margin)
        margin = 10
        x_offset = random.randint(margin, bg_width - new_width - margin)
        y_offset = random.randint(margin, bg_height - new_height - margin)
        
        # Overlay the image
        background = overlay_image(background, foreground_resized, x_offset, y_offset)
        
        # Calculate YOLO format bounding box
        x_center, y_center, width, height = convert_to_yolo_format(x_offset, y_offset, new_width, new_height, bg_width, bg_height)
        annotations.append(f"0 {x_center} {y_center} {width} {height}")
    
    # Save the result image
    output_path = os.path.join(output_dir, os.path.basename(background_path))
    cv2.imwrite(output_path, background)
    
    # Save the annotations
    label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(background_path))[0] + ".txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(annotations))
    
    print(f"Processed: {os.path.basename(background_path)} with {num_overlays} overlays")

# Set your directories
foreground_dir = 'D:/projects/img_blend/cropped_images'
# C:\Users\nazal\Downloads\pestvision_data\background_data\RiceLeafs\val\normal
background_dir = 'C:/Users/nazal/Downloads/pestvision_data/background_data/paddy_disease_classification/val/normal'
output_dir = 'D:/projects/img_blend/output_images'
labels_dir = 'D:/projects/img_blend/labels'

# Set number of background images to process
num_backgrounds_to_process = 50  # Change this to your desired number

# Create output and label directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Get list of foreground and background images
foreground_images = [os.path.join(foreground_dir, f) for f in os.listdir(foreground_dir) if f.endswith('.png') or f.endswith('.jpg')]
background_images = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.endswith('.jpg')]

# Randomly select the specified number of background images
selected_backgrounds = random.sample(background_images, min(num_backgrounds_to_process, len(background_images)))

# Process each selected background image
for bg_path in selected_backgrounds:
    process_image(bg_path, foreground_images, output_dir, labels_dir)

print(f"Processed {len(selected_backgrounds)} background images.")
