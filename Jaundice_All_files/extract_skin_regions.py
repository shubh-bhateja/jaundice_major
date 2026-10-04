import cv2
import numpy as np
import os

def extract_skin_region(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range for white color (the inner rectangle border)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No white region found in {image_path}")
        return
    # Find the largest white contour (should be the inner rectangle)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Slightly shrink the bounding box to avoid white border
    shrink = 5
    x1 = x + shrink
    y1 = y + shrink
    x2 = x + w - shrink
    y2 = y + h - shrink
    # Check for valid crop coordinates
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid crop region for {image_path}")
        return
    skin_crop = img[y1:y2, x1:x2]
    # Check if crop is empty
    if skin_crop.size == 0:
        print(f"Empty crop for {image_path}")
        return
    cv2.imwrite(output_path, skin_crop)


def process_all_images(input_folder, output_folder):
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            extract_skin_region(in_path, out_path)


if __name__ == "__main__":
    input_folder = "images"
    output_folder = "skin images"
    os.makedirs(output_folder, exist_ok=True)
    process_all_images(input_folder, output_folder)
    print("Skin region extraction complete.")
