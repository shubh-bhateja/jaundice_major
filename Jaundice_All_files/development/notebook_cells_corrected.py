"""
notebook_cells_corrected.py  (v2 — Box 1/Box 2 fix)
=====================================================
INSTRUCTIONS:
  In 1dcnn_paper_implementation.ipynb, REPLACE the entire source of
  Cell 3 (preprocessing functions) with everything below the dotted line.
  Then re-run Cells 3 → 4 → 5 → 6 → 7 in order.

─────────────────────────────────────────────────────────────────────────
"""

# ════════════════════════════════════════════════════════════════════
# [REPLACE: Cell 3] — Corrected Preprocessing Functions
# ════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import os

# ── Image structure:
#   Box 2 — small bright-yellow square (top area), used for white balance only
#   Box 1 — large rounded central window showing neonatal skin (the ROI)

# ── HSV ranges ────────────────────────────────────────────────────────
YELLOW_LOWER = np.array([22, 150, 150], dtype=np.uint8)
YELLOW_UPPER = np.array([38, 255, 255], dtype=np.uint8)

SKIN_LOWER_1 = np.array([0,   20,  60], dtype=np.uint8)   # warm skin
SKIN_UPPER_1 = np.array([25, 155, 255], dtype=np.uint8)
SKIN_LOWER_2 = np.array([160,  20,  60], dtype=np.uint8)  # wrap-around red
SKIN_UPPER_2 = np.array([179, 155, 255], dtype=np.uint8)

MIN_YELLOW_AREA = 2000
MIN_SKIN_AREA   = 10000
TARGET_PIXELS   = 4096   # fixed 1D sequence length


def detect_yellow_patch(image_bgr):
    """Find the yellow calibration patch (Box 2). Returns patch mask or None."""
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_YELLOW_AREA]
    if not valid:
        return None
    patch_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(patch_mask, [max(valid, key=cv2.contourArea)],
                     -1, 255, cv2.FILLED)
    return patch_mask


def detect_skin_roi_mask(image_bgr):
    """
    Find the neonatal skin window (Box 1) in the RAW image.

    IMPORTANT: always run on the original, un-white-balanced image.
    The LAB white balance shifts hue and would break this segmentation.
    """
    hsv   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.bitwise_or(cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1),
                           cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2))
    k_c   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_o   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_o)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_SKIN_AREA]
    if not valid:
        return None
    skin_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(skin_mask, [max(valid, key=cv2.contourArea)],
                     -1, 255, cv2.FILLED)
    return skin_mask


def lab_white_balance(image_bgr, yellow_patch_mask):
    """White-balance using the yellow patch. Returns corrected BGR image."""
    if yellow_patch_mask is None or np.sum(yellow_patch_mask) == 0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a = np.clip(a + (128.0 - float(cv2.mean(a, mask=yellow_patch_mask)[0])), 0, 255)
    b = np.clip(b + (128.0 - float(cv2.mean(b, mask=yellow_patch_mask)[0])), 0, 255)
    return cv2.cvtColor(cv2.merge([l, a, b]).astype(np.uint8), cv2.COLOR_LAB2BGR)


def extract_color_features(image_bgr, skin_mask):
    """Extract [R,G,B,H,S,V]/255 for skin pixels. Returns (N,6) float32."""
    if skin_mask is None:
        return None
    coords = np.where(skin_mask > 0)
    if len(coords[0]) == 0:
        return None
    rgb_px = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[coords[0], coords[1]]
    hsv_px = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[coords[0], coords[1]]
    return np.concatenate([rgb_px, hsv_px], axis=1).astype(np.float32) / 255.0


def make_fixed_sequence(features, target_pixels=TARGET_PIXELS):
    """Subsample or zero-pad to (target_pixels, 6)."""
    if features is None:
        return np.zeros((target_pixels, 6), dtype=np.float32)
    n = features.shape[0]
    if n >= target_pixels:
        return features[np.linspace(0, n-1, target_pixels, dtype=int)]
    return np.concatenate([features,
                           np.zeros((target_pixels-n, 6), np.float32)], axis=0)


def preprocess_image(path, img_size=None, target_pixels=TARGET_PIXELS):
    """
    Full corrected pipeline  (img_size ignored — always uses native 567×567).

    Order matters:
      1. Detect skin mask on RAW image   (before white balance!)
      2. White balance using yellow patch
      3. Extract features from balanced image using raw-detected skin mask
    """
    image     = cv2.imread(path)
    skin_mask = detect_skin_roi_mask(image)                  # ← raw image
    balanced  = lab_white_balance(image, detect_yellow_patch(image))
    features  = extract_color_features(balanced, skin_mask)  # ← balanced pixels
    return make_fixed_sequence(features, target_pixels)


# ── Quick verification (run this cell after replacing Cell 3) ─────────
sample_path = os.path.join(IMAGE_DIR, '0003-1.jpg')   # IMAGE_DIR defined in Cell 2
seq = preprocess_image(sample_path)
nz  = int(np.sum(seq.sum(axis=1) > 0))
print(f"Shape        : {seq.shape}   (expected: ({TARGET_PIXELS}, 6))")
print(f"Value range  : [{seq.min():.3f}, {seq.max():.3f}]   (expected: [0.0, 1.0])")
print(f"Non-zero rows: {nz}/{TARGET_PIXELS} ({nz/TARGET_PIXELS*100:.1f}%)")
print()
print("PASS ✅" if (nz > 3000 and seq.max() <= 1.0) else "FAIL ❌ — check HSV thresholds")
