import os
from PIL import Image

def convert_png_to_jpg(folder_path, log_file):
    # Create a log file to save the names of the changed files
    with open(log_file, 'w') as log:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                # Define the full path to the file
                png_path = os.path.join(folder_path, filename)
                # Define the new jpg path
                jpg_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.jpg')
                
                # Open the PNG image
                with Image.open(png_path) as img:
                    # Convert the image to RGB (PNG may have alpha channel)
                    rgb_img = img.convert('RGB')
                    # Save the image as JPG
                    rgb_img.save(jpg_path, 'JPEG')
                
                # Log the conversion
                log.write(f"{jpg_path},{png_path}\n")
                
                # Optionally, delete the original PNG file
                # os.remove(png_path)
                print(f"Converted {filename} to {os.path.basename(jpg_path)}")

def revert_jpg_to_png(log_file):
    # Read the log file to get the names of the changed files
    with open(log_file, 'r') as log:
        for line in log:
            jpg_path, png_path = line.strip().split(',')
            # Rename the jpg file back to png
            if os.path.exists(jpg_path):
                os.rename(jpg_path, png_path)
                print(f"Reverted {os.path.basename(jpg_path)} to {os.path.basename(png_path)}")

# Define the folder path and log file path
folder_path = "D:/projects/img_blend/output_images"
log_file = "conversion_log.txt"

# Convert PNG to JPG
convert_png_to_jpg(folder_path, log_file)

# To revert the changes, call the revert_jpg_to_png function
# revert_jpg_to_png(log_file)
