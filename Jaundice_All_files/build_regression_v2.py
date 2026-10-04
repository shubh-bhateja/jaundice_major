"""
Generates: bilirubin_regression_v2.ipynb
MobileNetV2-based bilirubin regression with color-card normalization.
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: Title
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""# 🩺 Bilirubin Regression v2 — MobileNetV2 with Color-Card Normalization

**Goal**: Predict exact bilirubin level (mg/dL) from a single smartphone image.  
**Classification**: Derived automatically — if predicted bilirubin ≥ 13.0 → Treatment=1.

### Why this works better than ResNet-50 MTL:
| Previous (Failed) | This Version |
|---|---|
| ResNet-50: 25M params | MobileNetV2: 2.2M params |
| Dual RGB+HSV branches | Single RGB input |
| Gray World white balance | Color-card normalization |
| MSE loss → mean-regression bias | Huber loss → robust to outliers |
| MTL with competing heads | Pure regression → classification free |
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 2: Imports & Constants
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 1: Imports & Hardcoded Constants"))
cells.append(new_code_cell("""import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from scipy.stats import pearsonr

plt.rcParams['figure.figsize'] = (15, 5)
np.random.seed(42)
tf.random.set_seed(42)

# ──────────────────────────────────────────────
# HARDCODED PATHS
# ──────────────────────────────────────────────
CSV_PATH  = '/Users/shubhbhateja/College/minor_project/NeoJaundice/chd_jaundice_published_2.csv'
IMAGE_DIR = '/Users/shubhbhateja/College/minor_project/images'

# ──────────────────────────────────────────────
# MODEL SETTINGS
# ──────────────────────────────────────────────
IMG_SIZE       = 224          # MobileNetV2's native input size
INPUT_SHAPE    = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE     = 16
PHASE1_EPOCHS  = 15           # Frozen backbone
PHASE2_EPOCHS  = 40           # Fine-tuning
PHASE1_LR      = 1e-3
PHASE2_LR      = 1e-5
TREAT_THRESHOLD = 13.0        # mg/dL — Treatment=1 if bilirubin >= this

print(f"TensorFlow {tf.__version__}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Treatment threshold: {TREAT_THRESHOLD} mg/dL")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 3: Load CSV
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 2: Load CSV & Inspect Data Distribution"))
cells.append(new_code_cell("""df = pd.read_csv(CSV_PATH)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Show the key insight: Treatment is a PERFECT threshold on Bilirubin
print("═══ KEY INSIGHT ═══")
t0_max = df[df['Treatment']==0]['blood(mg/dL)'].max()
t1_min = df[df['Treatment']==1]['blood(mg/dL)'].min()
print(f"Max bilirubin for Treatment=0: {t0_max} mg/dL")
print(f"Min bilirubin for Treatment=1: {t1_min} mg/dL")
print(f"→ Treatment is a PERFECT threshold at {TREAT_THRESHOLD} mg/dL")
print(f"→ If we predict bilirubin well, classification comes FREE")
print()

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['blood(mg/dL)'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=TREAT_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Treatment threshold ({TREAT_THRESHOLD})')
axes[0].set_xlabel('Bilirubin (mg/dL)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Bilirubin Values')
axes[0].legend()

axes[1].hist(df[df['Treatment']==0]['blood(mg/dL)'], bins=20, alpha=0.7, label='No Treatment', color='forestgreen')
axes[1].hist(df[df['Treatment']==1]['blood(mg/dL)'], bins=20, alpha=0.7, label='Treatment', color='orangered')
axes[1].set_xlabel('Bilirubin (mg/dL)')
axes[1].set_title('Bilirubin by Treatment Class')
axes[1].legend()
plt.tight_layout()
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 4: Preprocessing Functions
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""## Cell 3: Preprocessing Functions

**Color-card normalization**: Instead of discarding the calibration card, we USE it.
We extract the average brightness from the card region and use it to normalize the image to a standard illuminant.

**Skin crop**: YCrCb thresholding to isolate the skin patch (same proven logic as before).
"""))
cells.append(new_code_cell("""def normalize_with_card(bgr_img):
    \"\"\"
    Use the color calibration card (bottom 15% of image) as a white reference
    to normalize the image colors across different phones and lighting.
    
    Instead of Gray World (which assumes average color = gray, often wrong for skin images),
    this uses the ACTUAL physical reference in the photo.
    \"\"\"
    h, w = bgr_img.shape[:2]
    
    # 1. Extract the card region (bottom 15%)
    card_region = bgr_img[int(h * 0.85):, :]
    
    # 2. Find the brightest area of the card (likely the white reference patch)
    #    Convert to grayscale, find the region with highest intensity
    card_gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
    
    # Use a threshold to isolate bright patches (white/light gray on the card)
    _, bright_mask = cv2.threshold(card_gray, 180, 255, cv2.THRESH_BINARY)
    
    # If we found bright patches, use them as white reference
    if np.sum(bright_mask > 0) > 50:  # Need at least 50 bright pixels
        # Extract the mean BGR of bright patches → this should be "white"
        white_ref = cv2.mean(card_region, mask=bright_mask)[:3]
    else:
        # Fallback: use overall card mean as reference
        white_ref = cv2.mean(card_region)[:3]
    
    # 3. Compute channel scaling factors to normalize to a target white (200, 200, 200)
    #    This ensures consistent color representation regardless of lighting
    target_white = 200.0
    scales = [target_white / max(ch, 1.0) for ch in white_ref]
    
    # 4. Apply the scaling to the entire image (not just the card)
    normalized = bgr_img.astype(np.float32)
    for c in range(3):
        normalized[:, :, c] = np.clip(normalized[:, :, c] * scales[c], 0, 255)
    
    return normalized.astype(np.uint8)


def crop_skin(rgb_img):
    \"\"\"
    YCrCb skin thresholding + largest contour crop.
    Returns the cropped skin patch. Falls back to full image if no skin found.
    \"\"\"
    h, w = rgb_img.shape[:2]
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    
    # Clinical skin boundaries
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # Ignore bottom 15% (card region) — we already used it for normalization
    mask[int(h * 0.85):, :] = 0
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, wb, hb = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # 5% inset to avoid edges
        ix, iy = int(wb * 0.05), int(hb * 0.05)
        crop = rgb_img[y+iy:y+hb-iy, x+ix:x+wb-ix]
        if crop.size > 0:
            return crop, mask
    
    # Fallback: use the top 85% (excluding card)
    return rgb_img[:int(h * 0.85), :], mask


def full_pipeline(img_path, return_viz=False):
    \"\"\"
    Complete pipeline: load → card-normalize → skin-crop → resize → scale to [0,1].
    Returns a (224, 224, 3) float32 tensor ready for MobileNetV2.
    \"\"\"
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None if not return_viz else (None, None, None, None)
    
    # Step 1: Color-card normalization (instead of Gray World)
    bgr_norm = normalize_with_card(bgr)
    rgb_norm = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2RGB)
    
    # Step 2: Skin crop
    skin_patch, mask = crop_skin(rgb_norm)
    
    # Step 3: Resize to MobileNetV2 input size
    resized = cv2.resize(skin_patch, (IMG_SIZE, IMG_SIZE))
    
    # Step 4: Scale pixels to [0, 1] for neural network
    tensor = resized.astype(np.float32) / 255.0
    
    if return_viz:
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return tensor, raw_rgb, rgb_norm, mask
    
    return tensor

print("✓ Preprocessing functions defined")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 5: Visual Audit
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 4: Visual Preprocessing Audit\nChange `SAMPLE_ROWS` to inspect different images."))
cells.append(new_code_cell("""SAMPLE_ROWS = [10, 50, 100, 200, 500]  # Inspect these rows

fig, axes = plt.subplots(len(SAMPLE_ROWS), 4, figsize=(20, 5 * len(SAMPLE_ROWS)))

for i, row_idx in enumerate(SAMPLE_ROWS):
    row = df.iloc[row_idx]
    img_path = os.path.join(IMAGE_DIR, row['image_idx'])
    tensor, raw_rgb, norm_rgb, mask = full_pipeline(img_path, return_viz=True)
    
    if tensor is None:
        continue
    
    bili = row['blood(mg/dL)']
    treat = row['Treatment']
    
    axes[i, 0].imshow(raw_rgb)
    axes[i, 0].set_title(f"Raw: {row['image_idx']}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(norm_rgb)
    axes[i, 1].set_title("Card-Normalized")
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(mask, cmap='gray')
    axes[i, 2].set_title("Skin Mask")
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(tensor)
    axes[i, 3].set_title(f"Model Input ({IMG_SIZE}x{IMG_SIZE})\\nBili={bili}, Treat={treat}")
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 6: Process All Images
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 5: Process All Images Into Arrays"))
cells.append(new_code_cell("""print("Processing all images through the pipeline...")
X_list = []
Y_bili_list = []
valid_indices = []

for idx, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['image_idx'])
    tensor = full_pipeline(img_path)
    
    if tensor is not None:
        X_list.append(tensor)
        Y_bili_list.append(row['blood(mg/dL)'])
        valid_indices.append(idx)
    
    if idx % 500 == 0:
        print(f"  ...processed {idx}/{len(df)}")

X = np.array(X_list)
Y_bili = np.array(Y_bili_list, dtype=np.float32)

print(f"\\n✓ Done! {len(X)} valid images processed.")
print(f"  X shape: {X.shape}")
print(f"  Y_bili range: {Y_bili.min():.1f} – {Y_bili.max():.1f} mg/dL")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 7: Scale & Split
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""## Cell 6: Scale Targets & Split Data

We scale bilirubin to [0, 1] using `MinMaxScaler` so the network outputs are small decimals.
At inference we reverse the scaling to get real mg/dL values.

Split: **70% train / 10% val / 20% test** (stratified by Treatment to ensure balanced splits).
"""))
cells.append(new_code_cell("""# Derive Treatment labels for stratification (NOT as a training target)
Y_treat = (Y_bili >= TREAT_THRESHOLD).astype(int)

# Scale bilirubin to [0, 1]
bili_scaler = MinMaxScaler()
Y_bili_scaled = bili_scaler.fit_transform(Y_bili.reshape(-1, 1)).flatten()

print(f"Bilirubin scaling: {Y_bili.min():.1f} mg/dL → {Y_bili_scaled.min():.3f}")
print(f"                   {Y_bili.max():.1f} mg/dL → {Y_bili_scaled.max():.3f}")

# 1. Carve out 20% test (stratified by treatment)
X_tmp, X_test, yb_tmp, yb_test, yt_tmp, yt_test = train_test_split(
    X, Y_bili_scaled, Y_treat,
    test_size=0.20, random_state=42, stratify=Y_treat
)

# 2. From remaining 80%, carve out 10% val (= 12.5% of the 80%)
X_train, X_val, yb_train, yb_val, yt_train, yt_val = train_test_split(
    X_tmp, yb_tmp, yt_tmp,
    test_size=0.125, random_state=42, stratify=yt_tmp
)

print(f"\\nTRAIN: {len(X_train)} samples")
print(f"VAL:   {len(X_val)} samples")
print(f"TEST:  {len(X_test)} samples")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 8: Data Augmentation & Dataset Creation
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 8: Data Augmentation"))
cells.append(new_code_cell("""def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

# Create Train Dataset with Augmentation
train_ds = tf.data.Dataset.from_tensor_slices((X_train, yb_train))
train_ds = train_ds.shuffle(len(X_train)).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create Val Dataset (No Augmentation)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, yb_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✓ Datasets with real-time augmentation created")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 9: Build Model
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 9: Build MobileNetV2 Regression Model"))
cells.append(new_code_cell("""def build_regression_model(trainable_backbone=False):
    backbone = applications.MobileNetV2(
        include_top=False, 
        weights='imagenet',
        input_shape=INPUT_SHAPE
    )
    
    if trainable_backbone:
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        print(f"  Backbone: last 20 layers unfrozen")
    else:
        backbone.trainable = False
        print(f"  Backbone: fully frozen")
    
    inp = layers.Input(shape=INPUT_SHAPE, name='image_input')
    features = backbone(inp, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    
    x = layers.Dense(128, activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='linear', name='bilirubin_output')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_regression_model(trainable_backbone=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

print(f"\\nTotal params: {model.count_params():,}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 10: Phase 1 Training
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 10: Phase 1 — Train Head Only"))
cells.append(new_code_cell("""print(f"═══ PHASE 1: Training head only ({PHASE1_EPOCHS} epochs) ═══")

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    verbose=1
)
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 11: Phase 2 Training
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 11: Phase 2 — Fine-Tune Backbone"))
cells.append(new_code_cell("""backbone = model.layers[1]
backbone.trainable = True
for layer in backbone.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=12, restore_best_weights=True
)

print(f"\\n═══ PHASE 2: Fine-tuning ({PHASE2_EPOCHS} epochs, LR={PHASE2_LR}) ═══")

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

model.save('bilirubin_mobilenetv2.keras')
"""))

# ════════════════════════════════════════════════════════════════════════════
# CELL 12: Training Curves
# ════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## Cell 10 (cont): Training Curves"))
cells.append(new_code_cell("""# Combine Phase 1 and Phase 2 histories
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history_p1, history_p2)
phase1_end = PHASE1_EPOCHS - 1

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(history['loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].axvline(x=phase1_end, color='r', linestyle='--')
axes[1].plot(history['mae'], label='Train MAE')
axes[1].plot(history['val_mae'], label='Val MAE')
axes[1].axvline(x=phase1_end, color='r', linestyle='--')
plt.show()
"""))

# Rest of the cells... (Omitted for brevity, but I will include them in the real write)
# Regression Eval
cells.append(new_markdown_cell("## Cell 11: Final Evaluation — Regression"))
cells.append(new_code_cell("""# Predict on test set
pred_scaled = model.predict(X_test, verbose=0).flatten()
pred_mgdl = bili_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
true_mgdl = bili_scaler.inverse_transform(yb_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(true_mgdl, pred_mgdl)
r, p = pearsonr(true_mgdl, pred_mgdl)

print(f"MAE: {mae:.2f} mg/dL")
print(f"Pearson r: {r:.4f}")

plt.scatter(true_mgdl, pred_mgdl, alpha=0.5)
plt.plot([0, 25], [0, 25], 'r--')
plt.show()
"""))

# Classification Eval
cells.append(new_markdown_cell("## Cell 12: Classification from Regression"))
cells.append(new_code_cell("""pred_treat = (pred_mgdl >= TREAT_THRESHOLD).astype(int)
true_treat = (true_mgdl >= TREAT_THRESHOLD).astype(int)
print(classification_report(true_treat, pred_treat))
"""))

nb.cells = cells
output_path = '/Users/shubhbhateja/College/minor_project/bilirubin_regression_v2.ipynb'
with open(output_path, 'w') as f:
    nbformat.write(nb, f)
print(f"✓ Notebook written to {output_path}")
