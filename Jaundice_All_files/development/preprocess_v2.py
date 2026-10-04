"""
preprocess_v2.py — Corrected NeoJaundice Image Preprocessing Pipeline
=======================================================================
Paper: Makhloughi (2025), Scientific Reports

Image structure (confirmed from dataset analysis):
  - Box 2 (yellow patch, ~136×136 px, top area of calibration card):
      Bright-yellow square. Used ONLY for LAB white balancing.
  - Box 1 (skin window, ~270×270 px, centre of frame):
      Large rounded central window showing the neonate's skin.
      This is the skin ROI we extract features from.

Key design decision:
  - Skin region is detected on the RAW image (before white balance),
    because the LAB white balance can shift hue and break segmentation.
  - White balance is applied to the extracted skin pixels only for
    colour normalisation before feature extraction.
"""

import cv2
import numpy as np

# ─── HSV Constants ────────────────────────────────────────────────────────────

# Box 2: pure bright yellow patch
YELLOW_LOWER = np.array([22, 150, 150], dtype=np.uint8)
YELLOW_UPPER = np.array([38, 255, 255], dtype=np.uint8)

# Box 1: neonatal skin tone (H=0-25 + H=160-179 to handle wrap-around reds)
SKIN_LOWER_1  = np.array([0,   20, 60],  dtype=np.uint8)
SKIN_UPPER_1  = np.array([25, 155, 255], dtype=np.uint8)
SKIN_LOWER_2  = np.array([160, 20,  60], dtype=np.uint8)   # wrap-around red
SKIN_UPPER_2  = np.array([179, 155, 255], dtype=np.uint8)

MIN_YELLOW_AREA = 2000   # px²
MIN_SKIN_AREA   = 10000  # px²

TARGET_PIXELS = 4096


# ─── Step 1: Load at full resolution ─────────────────────────────────────────

def load_full_res(path: str) -> np.ndarray:
    """Load BGR image at native 567×567 resolution."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


# ─── Step 2: Detect yellow reference patch (Box 2) ───────────────────────────

def detect_yellow_patch(image_bgr: np.ndarray):
    """
    Find the bright-yellow calibration patch (Box 2).
    Returns (raw_yellow_mask, clean_patch_mask).
    patch_mask is None if detection fails.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_YELLOW_AREA]
    if not valid:
        return mask, None

    largest = max(valid, key=cv2.contourArea)
    patch_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(patch_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask, patch_mask


# ─── Step 3: Detect skin window (Box 1) on RAW image ─────────────────────────

def detect_skin_roi_mask(image_bgr: np.ndarray):
    """
    Segment the neonatal skin window (Box 1) from the RAW image.

    Works on the original BGR before any white balance because the LAB
    white balance can shift hue drastically, breaking HSV thresholding.

    Combines two HSV ranges to handle hue wrap-around (red ≈ H 160-179).
    Returns a binary mask (uint8), or None if no skin region found.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1)
    mask2 = cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup: close small holes, remove thin noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_SKIN_AREA]
    if not valid:
        return None

    # Largest blob = skin window (>> any card patch)
    largest = max(valid, key=cv2.contourArea)

    skin_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(skin_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return skin_mask


# ─── Step 4: LAB white balance (applied to full image for pixel-level use) ───

def lab_white_balance(image_bgr: np.ndarray, yellow_patch_mask) -> np.ndarray:
    """
    White-balance the image using mean LAB of the yellow calibration patch.
    Shifts A and B channels toward neutral (128, 128).
    """
    if yellow_patch_mask is None or np.sum(yellow_patch_mask) == 0:
        return image_bgr
    # Ensure mask is uint8
    pm = yellow_patch_mask.astype(np.uint8)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    # Use masked mean on float arrays — compute manually to avoid OpenCV type issues
    card_a = float(a[pm > 0].mean()) if pm.sum() > 0 else 128.0
    card_b = float(b[pm > 0].mean()) if pm.sum() > 0 else 128.0
    a = np.clip(a + (128.0 - card_a), 0, 255)
    b = np.clip(b + (128.0 - card_b), 0, 255)

    balanced = cv2.merge([l, a, b]).astype(np.uint8)
    return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)


# ─── Step 5: Extract RGB+HSV features from skin pixels ───────────────────────

def extract_color_features(image_bgr: np.ndarray, skin_mask: np.ndarray):
    """
    Extract [R, G, B, H, S, V] for every pixel inside the skin mask.
    image_bgr here should be the WHITE-BALANCED image for colour accuracy.
    Returns (N, 6) float32 in [0,1], or None if no pixels.
    """
    if skin_mask is None:
        return None

    coords = np.where(skin_mask > 0)
    if len(coords[0]) == 0:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    rgb_px = image_rgb[coords[0], coords[1], :]
    hsv_px = image_hsv[coords[0], coords[1], :]

    features = np.concatenate([rgb_px, hsv_px], axis=1).astype(np.float32)
    features /= 255.0
    return features   # (N, 6)


# ─── Step 6: Fixed-length 1D sequence ────────────────────────────────────────

def make_fixed_sequence(features, target_pixels: int = TARGET_PIXELS):
    """Uniform-subsample or zero-pad to (target_pixels, 6)."""
    if features is None:
        return np.zeros((target_pixels, 6), dtype=np.float32)

    n = features.shape[0]
    if n >= target_pixels:
        idx = np.linspace(0, n - 1, target_pixels, dtype=int)
        return features[idx]

    pad = np.zeros((target_pixels - n, 6), dtype=np.float32)
    return np.concatenate([features, pad], axis=0)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def preprocess_image(path: str, img_size=None, target_pixels: int = TARGET_PIXELS):
    """
    Complete corrected preprocessing pipeline.

    1. Load at full resolution
    2. Detect skin window mask (Box 1) on RAW image   ← avoids WB hue shift
    3. Detect yellow patch (Box 2) for white balance
    4. Apply LAB white balance to full image
    5. Extract RGB+HSV features from skin pixels of white-balanced image
    6. Build fixed (target_pixels, 6) 1D array

    img_size is accepted for API compatibility but ignored.
    """
    image = load_full_res(path)

    # Step 2 FIRST — detect skin on raw image
    skin_mask = detect_skin_roi_mask(image)

    # Step 3-4 — white balance using yellow patch
    _, yellow_patch_mask = detect_yellow_patch(image)
    balanced = lab_white_balance(image, yellow_patch_mask)

    # Step 5-6 — features from white-balanced image, using raw-detected mask
    features = extract_color_features(balanced, skin_mask)
    return make_fixed_sequence(features, target_pixels)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, time

    IMAGE_DIR = "/Users/shubhbhateja/College/minor_project/images"
    test_imgs = ["0001-1.jpg", "0005-1.jpg", "0010-1.jpg", "0003-1.jpg", "0002-1.jpg"]

    print(f"{'Image':<15} {'Skin px':>10}  {'Non-zero':>10}  {'Range':>14}  {'ms':>6}")
    print("-" * 65)

    for fname in test_imgs:
        path = os.path.join(IMAGE_DIR, fname)
        t0 = time.time()
        seq = preprocess_image(path)
        ms = (time.time() - t0) * 1000

        nz = int(np.sum(seq.sum(axis=1) > 0))

        # Count raw skin pixels for info
        img_raw = load_full_res(path)
        sm = detect_skin_roi_mask(img_raw)
        n_skin = int(np.sum(sm > 0)) if sm is not None else 0

        print(f"{fname:<15} {n_skin:>10,}  {nz:>10,}  "
              f"[{seq.min():.3f}, {seq.max():.3f}]  {ms:>6.0f}")
