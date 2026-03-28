import os
import shutil
import cv2
import numpy as np

images_folder = "CrackForest/Images"
masks_folder = "CrackForest/Masks"

output_crack = "dataset/crack"
output_no_crack = "dataset/no_crack"

os.makedirs(output_crack, exist_ok=True)
os.makedirs(output_no_crack, exist_ok=True)

for filename in os.listdir(images_folder):
    image_path = os.path.join(images_folder, filename)

    # Example: 110.jpg -> 110
    name = os.path.splitext(filename)[0]

    # CrackForest mask format: 110_label.PNG
    mask_name = name + "_label.PNG"
    mask_path = os.path.join(masks_folder, mask_name)

    if not os.path.exists(mask_path):
        print(f"Mask not found for {filename}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not read mask for {filename}")
        continue

    # Count crack pixels
    white_pixels = np.sum(mask > 10)

    if white_pixels > 100:
        shutil.copy(image_path, os.path.join(output_crack, filename))
    else:
        shutil.copy(image_path, os.path.join(output_no_crack, filename))

print("Dataset preparation complete.")