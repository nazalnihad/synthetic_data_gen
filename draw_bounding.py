import cv2
import os
import random

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    height, width = image.shape[:2]
    x_center, y_center, box_width, box_height = bbox
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_bounding_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            draw_bounding_box(image, (x_center, y_center, box_width, box_height))
    return image

def process_random_images_with_bboxes(images_dir, labels_dir, output_dir, num_images):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    if num_images > len(image_files):
        print(f"Requested number of images ({num_images}) exceeds available images ({len(image_files)}). Processing all available images.")
        num_images = len(image_files)

    random_sample = random.sample(image_files, num_images)

    for image_file in random_sample:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
        if os.path.exists(label_path):
            image_with_bboxes = draw_bounding_boxes(image_path, label_path)
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image_with_bboxes)
            print(f"Processed and saved: {output_path}")

# Set your directories
images_dir = 'D:/projects/img_blend/output_images'
labels_dir = 'D:/projects/img_blend/output_images'
output_dir = 'D:/projects/img_blend/bbox_images'
num_images = 50  # Set the number of random images to process

# Process random images and draw bounding boxes
process_random_images_with_bboxes(images_dir, labels_dir, output_dir, num_images)

print(f"Processed {num_images} images with bounding boxes saved to: {output_dir}")
