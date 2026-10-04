import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    return image

def detect_calibration_card(image_bgr):
    """
    1. Load an image and convert it from BGR to HSV.
    2. Define a yellow color range to mask and detect the color calibration card.
    Adjust the lower and upper bounds depending on lighting conditions.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Define yellow color range in HSV space
    # In OpenCV H is 0-179, S is 0-255, V is 0-255. Yellow is roughly H=20-35.
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Mask yellow regions
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    
    return mask_yellow

def perform_white_balancing(image_bgr, mask_yellow):
    """
    3. Use the average LAB values of the card to perform White Balancing on the whole image.
    Calculates the required shift based on the card's average LAB vs a target LAB.
    """
    # Convert image to LAB color space
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    # Extract the LAB values of the calibration card using the mask
    mean_lab = cv2.mean(image_lab, mask=mask_yellow)
    mean_l, mean_a, mean_b = mean_lab[0], mean_lab[1], mean_lab[2]
    
    if np.all(mean_lab[:3] == 0):
        print("Warning: Calibration card not detected. Returning original image.")
        return image_bgr
        
    # Standard white balancing adjust A and B channels towards neutral or a known reference point.
    # Note: Replace TARGET_L, TARGET_A, TARGET_B with your actual reference calibration values.
    # For a theoretical perfectly neutral grey we use A=128, B=128. Since it is yellow, we adjust
    # according to its target reference values.
    TARGET_L = 200.0
    TARGET_A = 120.0  
    TARGET_B = 170.0  
    
    # Calculate difference
    shift_l = TARGET_L - mean_l
    shift_a = TARGET_A - mean_a
    shift_b = TARGET_B - mean_b
    
    # Split channels, apply shift, and ensure values remain in valid [0, 255] range
    l, a, b = cv2.split(image_lab)
    l_balanced = cv2.convertScaleAbs(l, alpha=1, beta=shift_l)
    a_balanced = cv2.convertScaleAbs(a, alpha=1, beta=shift_a)
    b_balanced = cv2.convertScaleAbs(b, alpha=1, beta=shift_b)
    
    # Merge channels
    balanced_lab = cv2.merge((l_balanced, a_balanced, b_balanced))
    
    # Convert the balanced image back to BGR
    balanced_bgr = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)
    
    return balanced_bgr

def segment_skin_roi(image_bgr):
    """
    4. Isolate the Skin ROI. 
    Here we use YCrCb color space segmentation as it isolates luminance. 
    You might adjust these values based on the dataset if necessary.
    """
    image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Skin color bounds in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(image_ycrcb, lower_skin, upper_skin)
    
    # Extract skin ROI using the generated mask
    skin_roi = cv2.bitwise_and(image_bgr, image_bgr, mask=skin_mask)
    
    return skin_roi, skin_mask

def convert_to_1d_sequence(skin_roi, skin_mask):
    """
    Converts segmented pixels into a 1D sequence for 1DCNN input.
    """
    # Extract only the valid skin pixels (where the mask > 0)
    skin_pixels = skin_roi[skin_mask > 0]
    
    # Convert into a 1D sequence. 
    # skin_pixels is (N, 3), flatten() will make it (N * 3,)
    sequence_1d = skin_pixels.flatten()
    
    return sequence_1d

def process_pipeline(image_path):
    # Load Image
    image_bgr = load_image(image_path)
    
    # Detect calibration card (HSV thresholding)
    mask_yellow = detect_calibration_card(image_bgr)
    
    # White Balancing (LAB space)
    balanced_bgr = perform_white_balancing(image_bgr, mask_yellow)
    
    # Segment Skin ROI (YCrCb space thresholding)
    skin_roi, skin_mask = segment_skin_roi(balanced_bgr)
    
    # Convert to 1D sequence for 1DCNN
    sequence_1d = convert_to_1d_sequence(skin_roi, skin_mask)
    
    # Output Info
    print(f"Original shape: {image_bgr.shape}")
    print(f"Skin pixels extracted: {np.count_nonzero(skin_mask)}")
    print(f"1D sequence shape for 1DCNN: {sequence_1d.shape}")
    
    return sequence_1d

if __name__ == "__main__":
    # Test execution
    # Replace 'sample_image.jpg' with a real image to test the code
    # sequence = process_pipeline('sample_image.jpg')
    print("Pre-processing script is ready to run.")
