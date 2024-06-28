import cv2
import numpy as np
import os
import random

def crop_and_save_image(mask_path, img_path, output_dir):
    # Load the images
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
    if len(y) > 0 and len(x) > 0:
        top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
        cropped_img = img_rgba[top:bottom+1, left:right+1]

        # Save the cropped image
        output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '_cropped.png'))
        cv2.imwrite(output_path, cropped_img)
        return True
    return False

def process_images(mask_dir, img_dir, output_dir, n):
    # Get list of mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Shuffle the list to get random selection
    random.shuffle(mask_files)
    
    # Process n images
    processed = 0
    for mask_file in mask_files:
        if processed >= n:
            break
        
        mask_path = os.path.join(mask_dir, mask_file)
        img_file = mask_file.replace('.png', '.jpg')
        img_path = os.path.join(img_dir, img_file)
        
        if os.path.exists(img_path):
            if crop_and_save_image(mask_path, img_path, output_dir):
                processed += 1
                print(f"Processed {processed}/{n}: {img_file}")

    print(f"Finished processing {processed} images.")

# Set your directories and number of images to process
mask_dir = 'C:/Users/nazal/Downloads/pestvision_data/foreground_data/Detection_IP102/Masks'
img_dir = 'C:/Users/nazal/Downloads/pestvision_data/foreground_data/Detection_IP102/JPEGImages'
output_dir = 'cropped_images'
n = 8000  # Number of images to process

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process the images
process_images(mask_dir, img_dir, output_dir, n)