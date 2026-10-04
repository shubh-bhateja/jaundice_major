"""
preprocess_diagnostic.py — Verification script for the fixed pipeline
=======================================================================
Runs the v2 preprocessing on 10 sample images and confirms:
  ✅ Card detected for each image
  ✅ Skin pixel count > 500 per image
  ✅ Feature values in [0, 1]
  ✅ No all-zeros output (the bug from v1)

Also compares v1 (broken) vs v2 (fixed) non-zero pixel counts side by side.
"""

import os, sys, time
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from preprocess_v2 import (
    preprocess_image, detect_card_bbox, extract_skin_roi,
    load_full_res, lab_white_balance, YELLOW_LOWER, YELLOW_UPPER, TARGET_PIXELS
)

IMAGE_DIR = "/Users/shubhbhateja/College/minor_project/images"
SAMPLE_IMGS = sorted(os.listdir(IMAGE_DIR))[:10]   # first 10 images


# ─── Old v1 pipeline (for comparison) ────────────────────────────────────────

def preprocess_v1_broken(path, img_size=64, target_pixels=TARGET_PIXELS):
    """The original broken pipeline (resize first, then detect card)."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))     # 64×64 — destroys card geometry

    # LAB white balance on the resized image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    avg_a, avg_b = np.mean(a), np.mean(b)
    l_float = l.astype(np.float32) / 255.0
    a = np.clip(a.astype(np.float32) - (avg_a - 128) * l_float, 0, 255)
    b = np.clip(b.astype(np.float32) - (avg_b - 128) * l_float, 0, 255)
    lab = cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # HSV skin mask on resized image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))

    skin_coords = np.where(mask > 0)
    if len(skin_coords[0]) == 0:
        return np.zeros((target_pixels, 6), dtype=np.float32)

    rgb_px = image[skin_coords[0], skin_coords[1], :]
    hsv_px = hsv[skin_coords[0], skin_coords[1], :]
    combined = np.concatenate([rgb_px, hsv_px], axis=1).astype(np.float32) / 255.0

    n = combined.shape[0]
    if n >= target_pixels:
        indices = np.linspace(0, n-1, target_pixels, dtype=int)
        return combined[indices]
    pad = np.zeros((target_pixels - n, 6), dtype=np.float32)
    return np.concatenate([combined, pad], axis=0)


# ─── Run comparison ───────────────────────────────────────────────────────────

print("=" * 65)
print("NeoJaundice Preprocessing Diagnostic: v1 (broken) vs v2 (fixed)")
print("=" * 65)
print(f"{'Image':<18} {'v1 non-zero':>12} {'v2 non-zero':>12}  {'Card?':>6}  {'v2 ok?':>7}")
print("-" * 65)

results = []
all_pass = True

for fname in SAMPLE_IMGS:
    path = os.path.join(IMAGE_DIR, fname)

    seq_v1 = preprocess_v1_broken(path)
    seq_v2 = preprocess_image(path)

    nz_v1 = int(np.sum(seq_v1.sum(axis=1) > 0))
    nz_v2 = int(np.sum(seq_v2.sum(axis=1) > 0))

    # Card detection check
    img = load_full_res(path)
    card_bbox, _ = detect_card_bbox(img)
    card_ok = "✅" if card_bbox is not None else "❌"

    # v2 quality checks
    v2_ok = (
        nz_v2 > 500 and
        seq_v2.min() >= 0.0 and
        seq_v2.max() <= 1.0
    )
    v2_str = "✅" if v2_ok else "❌"
    if not v2_ok:
        all_pass = False

    print(f"{fname:<18} {nz_v1:>12,} {nz_v2:>12,}  {card_ok:>6}  {v2_str:>7}")
    results.append((fname, nz_v1, nz_v2, card_bbox is not None))

print("=" * 65)
print(f"\nOverall result: {'✅ ALL PASS' if all_pass else '❌ SOME FAILURES'}")

# Average improvement
avg_v1 = np.mean([r[1] for r in results])
avg_v2 = np.mean([r[2] for r in results])
print(f"\nAvg non-zero rows — v1: {avg_v1:.0f} ({avg_v1/TARGET_PIXELS*100:.1f}%)")
print(f"Avg non-zero rows — v2: {avg_v2:.0f} ({avg_v2/TARGET_PIXELS*100:.1f}%)")
print(f"Improvement: {avg_v2/max(avg_v1,1):.1f}× more real skin pixels extracted")


# ─── Visualise card detection on first 3 images ──────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
for i, fname in enumerate(SAMPLE_IMGS[:3]):
    path = os.path.join(IMAGE_DIR, fname)
    img_bgr = load_full_res(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    card_bbox, yellow_mask = detect_card_bbox(img_bgr)

    # Draw bounding box on a copy
    vis = img_rgb.copy()
    if card_bbox:
        x, y, bw, bh = card_bbox
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 4)
        # Draw the inner skin region
        balanced = lab_white_balance(img_bgr, yellow_mask)
        crop, skin_mask = extract_skin_roi(balanced, card_bbox)
        skin_only = cv2.bitwise_and(crop, crop, mask=skin_mask)
        skin_only_rgb = cv2.cvtColor(skin_only, cv2.COLOR_BGR2RGB)

    # Column 0: original + bounding box
    axes[i, 0].imshow(vis)
    axes[i, 0].set_title(f"{fname}\n(green = card bbox)", fontsize=9)
    axes[i, 0].axis('off')

    # Column 1: yellow mask
    axes[i, 1].imshow(yellow_mask, cmap='gray')
    axes[i, 1].set_title("Yellow card mask", fontsize=9)
    axes[i, 1].axis('off')

    # Column 2: extracted skin patch
    if card_bbox:
        axes[i, 2].imshow(skin_only_rgb)
        nz = np.sum(skin_mask > 0)
        axes[i, 2].set_title(f"Skin ROI ({nz} pixels)", fontsize=9)
    else:
        axes[i, 2].text(0.5, 0.5, "Card not detected", ha='center', va='center')
    axes[i, 2].axis('off')

plt.suptitle("v2 Preprocessing: Card Detection & Skin ROI Extraction", fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "diagnostic_visualization.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nVisualization saved to: {out_path}")
plt.show()
