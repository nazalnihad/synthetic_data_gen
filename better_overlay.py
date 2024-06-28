import os
import cv2
import numpy as np
import random
import math

def rotate(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def sobel_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    score_x = np.sum(np.abs(sobelx))
    score_y = np.sum(np.abs(sobely))
    return score_x, score_y

def find_optimal_angle(score_x, score_y):
    if score_x > score_y:
        return random.choice(range(0, 45)) if random.random() < 0.5 else random.choice(range(135, 180))
    else:
        return random.choice(range(45, 135))

def calculate_yolo_format(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2.0 / img_width
    y_center = (box[1] + box[3]) / 2.0 / img_height
    width = (box[2] - box[0]) / img_width
    height = (box[3] - box[1]) / img_height
    return [0, x_center, y_center, width, height]  # Class 0

def gather_all_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                images.append(os.path.join(root, file))
    return images

def overlay_images(background_folder, small_images_folder, output_folder, labels_folder, n, blur_prob=0.1, small_prob=0.05, pref='RDT'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    background_files = gather_all_images(background_folder)
    small_image_files = [f for f in os.listdir(small_images_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Ensure that n is not greater than the number of background images available
    if n > len(background_files):
        print(f"Warning: Requested number of background images ({n}) exceeds available images ({len(background_files)}). Adjusting to available images.")
        n = len(background_files)

    # background_files = random.sample(background_files, n)

    for i, background_file in enumerate(background_files):
        l_img = cv2.imread(background_file)

        l_height, l_width = l_img.shape[:2]
        max_overlay_size = min(l_width, l_height) // 8

        num_overlays = random.randint(1, 7)
        selected_small_images = random.sample(small_image_files, num_overlays)

        annotations = []

        for small_image_file in selected_small_images:
            small_image_path = os.path.join(small_images_folder, small_image_file)
            s_img = cv2.imread(small_image_path, cv2.IMREAD_UNCHANGED)

            s_height, s_width = s_img.shape[:2]

            if max(s_width, s_height) > max_overlay_size:
                scale = max_overlay_size / max(s_width, s_height)
                new_width = int(s_width * scale)
                new_height = int(s_height * scale)
                s_img_resized = cv2.resize(s_img, (new_width, new_height))
            else:
                s_img_resized = s_img.copy()

            score_x, score_y = sobel_score(l_img)
            optimal_angle = find_optimal_angle(score_x, score_y)

            s_img_rotated = rotate(s_img_resized, optimal_angle)

            if random.random() < blur_prob:
                s_img_rotated = cv2.GaussianBlur(s_img_rotated, (5, 5), 0)

            if random.random() < small_prob:
                if s_img_rotated.shape[1] // 2 > 0 and s_img_rotated.shape[0] // 2 > 0:
                    s_img_rotated = cv2.resize(s_img_rotated, (s_img_rotated.shape[1] // 2, s_img_rotated.shape[0] // 2))

            if s_img_rotated is None or s_img_rotated.shape[0] == 0 or s_img_rotated.shape[1] == 0:
                continue

            gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 1000
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            if valid_contours:
                contour = random.choice(valid_contours)
                point = contour[random.randint(0, len(contour) - 1)][0]
                x_offset = max(0, min(point[0] - s_img_rotated.shape[1] // 2, l_width - s_img_rotated.shape[1]))
                y_offset = max(0, min(point[1] - s_img_rotated.shape[0] // 2, l_height - s_img_rotated.shape[0]))
            else:
                x_offset = random.randint(0, l_width - s_img_rotated.shape[1])
                y_offset = random.randint(0, l_height - s_img_rotated.shape[0])

            box = [x_offset, y_offset, x_offset + s_img_rotated.shape[1], y_offset + s_img_rotated.shape[0]]
            yolo_annotation = calculate_yolo_format(box, l_width, l_height)
            annotations.append(yolo_annotation)

            if s_img_rotated.shape[2] == 4:
                alpha_s = s_img_rotated[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    l_img[y_offset:y_offset + s_img_rotated.shape[0], x_offset:x_offset + s_img_rotated.shape[1], c] = (
                            alpha_s * s_img_rotated[:, :, c] +
                            alpha_l * l_img[y_offset:y_offset + s_img_rotated.shape[0], x_offset:x_offset + s_img_rotated.shape[1], c]
                    )
            else:
                l_img[y_offset:y_offset + s_img_rotated.shape[0], x_offset:x_offset + s_img_rotated.shape[1]] = s_img_rotated

            # Apply a slight Gaussian blur to the whole image for blending
            l_img = cv2.GaussianBlur(l_img, (3, 3), 0.2)

        output_path = os.path.join(output_folder, f"{pref}_{i + 1}.png")
        cv2.imwrite(output_path, l_img)

        label_path = os.path.join(labels_folder, f"{pref}_{i + 1}.txt")
        with open(label_path, 'w') as f:
            for annotation in annotations:
                f.write(" ".join(map(str, annotation)) + "\n")

# Define the folder paths
background_folder = "C:/Users/nazal/Downloads/pestvision_data/background_data/RiceLeafs/train"
small_images_folder = "D:/projects/img_blend/cropped_images"
output_folder = "D:/projects/img_blend/output_images"
labels_folder = "D:/projects/img_blend/labels"

# Number of background images to process
n = 8000

# Run the overlay process
overlay_images(background_folder, small_images_folder, output_folder, labels_folder, n)
