import os
import random
import shutil

def copy_random_images_and_labels(data_folder, labels_folder, output_data_folder, output_labels_folder, n, prefix):
    # Ensure output folders exist
    os.makedirs(output_data_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # Get a list of all label files that start with the given prefix
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt') and f.startswith(prefix)]
    print(f"Total label files found with prefix '{prefix}': {len(label_files)}")

    # Ensure n is not larger than the number of available label files
    if n > len(label_files):
        print(f"Warning: Requested number of labels ({n}) exceeds available labels ({len(label_files)}). Adjusting to available labels.")
        n = len(label_files)

    # Randomly select n label files
    selected_label_files = random.sample(label_files, n)
    print(f"Selected {n} random label files with prefix '{prefix}'.")

    for label_file in selected_label_files:
        # Define the paths
        label_path = os.path.join(labels_folder, label_file)
        image_name = os.path.splitext(label_file)[0]  # Remove the .txt extension to get the image name

        # Define possible image extensions
        possible_extensions = ['.png', '.jpg', '.jpeg']
        image_path = None
        for ext in possible_extensions:
            potential_image_path = os.path.join(data_folder, image_name + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break

        # If the image exists, copy both image and label to the output folders
        if image_path:
            print(f"Copying image {os.path.basename(image_path)} and label {label_file}")
            shutil.copy(image_path, os.path.join(output_data_folder, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(output_labels_folder, os.path.basename(label_path)))
        else:
            print(f"Warning: No corresponding image found for label {label_file}")

# Define the folder paths
data_folder = "D:/projects/iist/backup_data/final/f_data/images/train"
labels_folder = "D:/projects/iist/backup_data/final/f_data/labels/train"
output_data_folder = "D:/projects/img_blend/output_images"
output_labels_folder = "D:/projects/img_blend/output_images"

# Number of random labels and corresponding images to copy
n = 8000

# Define the prefix to filter files
prefix = "IP"

# Run the process
copy_random_images_and_labels(data_folder, labels_folder, output_data_folder, output_labels_folder, n, prefix)
