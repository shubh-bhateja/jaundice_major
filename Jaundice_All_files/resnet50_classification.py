import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CSV_PATH = '/Users/shubhbhateja/College/minor_project/NeoJaundice/chd_jaundice_published_2.csv'
IMAGE_DIR = '/Users/shubhbhateja/College/minor_project/images'
INPUT_SIZE = (128, 128)

# --- PREPROCESSING FUNCTIONS (Preserving Logic from paper_classification_v1.ipynb) ---

def professional_white_balance(img):
    """Calibrates colors using Gray World algorithm before cropping."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def adaptive_crop_skin(image_rgb):
    """Isolates skin patch and removes borders/background."""
    h, w = image_rgb.shape[:2]
    img_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(img_ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # Clean noise
    mask[int(h*0.85):h, :] = 0 # Ignore bottom card area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return cv2.resize(image_rgb, INPUT_SIZE)
    
    x, y, w_b, h_b = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Inset crop to avoid edge shadows
    ix, iy = int(w_b*0.05), int(h_b*0.05)
    return image_rgb[y+iy:y+h_b-iy, x+ix:x+w_b-ix]

def prepare_image_2d(img_path):
    """
    Prepares images for Multimodal ResNet-50 (RGB and HSV 2D Tensors).
    Strictly following logic from paper_classification_v1.ipynb.
    """
    img = cv2.imread(img_path)
    if img is None: return None, None
    
    # 1. Calibrate on FULL image
    balanced = professional_white_balance(img)
    rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
    
    # 2. Crop Skin and Resize
    skin_rgb = adaptive_crop_skin(rgb)
    resized_rgb = cv2.resize(skin_rgb, INPUT_SIZE)
    
    # 3. Create HSV version
    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)
    
    # Normalize for ResNet (0-1)
    return resized_rgb / 255.0, hsv / 255.0

# --- MODEL BUILDING ---

def build_resnet50_model():
    """
    Builds a Multimodal Fusion model using Dual ResNet-50 branches.
    Branch 1: RGB Image
    Branch 2: HSV Image
    """
    # Define Inputs
    input_rgb = layers.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name='rgb_input')
    input_hsv = layers.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), name='hsv_input')

    # RGB Branch
    shared_resnet = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    shared_resnet.trainable = False

    # Process RGB
    x_rgb = shared_resnet(input_rgb)
    x_rgb = layers.GlobalAveragePooling2D()(x_rgb)

    # Process HSV
    x_hsv = shared_resnet(input_hsv)
    x_hsv = layers.GlobalAveragePooling2D()(x_hsv)

    # Fusion
    merged = layers.Concatenate()([x_rgb, x_hsv])
    
    # Dense Head
    x = layers.Dense(512, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name='treatment_output')(x)

    model = models.Model(inputs=[input_rgb, input_hsv], outputs=out)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- MAIN EXECUTION PIPELINE ---

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    X_rgb, X_hsv, Y = [], [], []
    
    print("Processing images for Multimodal ResNet-50...")
    for idx, row in df.iterrows():
        path = os.path.join(IMAGE_DIR, row['image_idx'])
        rgb_img, hsv_img = prepare_image_2d(path)
        if rgb_img is not None:
            X_rgb.append(rgb_img)
            X_hsv.append(hsv_img)
            Y.append(row['Treatment'])
        if idx % 200 == 0:
            print(f"Processed {idx}...")
            
    X_rgb = np.array(X_rgb)
    X_hsv = np.array(X_hsv)
    Y = np.array(Y)
    
    print(f"Total processed: {len(X_rgb)}")
    
    # Split data
    split = train_test_split(X_rgb, X_hsv, Y, test_size=0.2, random_state=42)
    X_rgb_tr, X_rgb_ts, X_hsv_tr, X_hsv_ts, y_tr, y_ts = split
    
    # Handle class imbalance
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    dict_weights = {0: weights[0], 1: weights[1]}
    
    # LR Scheduler
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 20: lr *= 0.1
        return lr
    
    # Build and Train
    model = build_resnet50_model()
    # print(model.summary())
    
    print("Starting Multimodal ResNet-50 classification training...")
    history = model.fit(
        x={'rgb_input': X_rgb_tr, 'hsv_input': X_hsv_tr},
        y=y_tr,
        validation_data=({'rgb_input': X_rgb_ts, 'hsv_input': X_hsv_ts}, y_ts),
        epochs=50,
        batch_size=16, # Reduced batch size for dual-resnet memory
        class_weight=dict_weights,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)],
        verbose=1
    )
    
    # Save model
    model.save('multimodal_resnet50_jaundice_model.h5')
    print("Model saved as multimodal_resnet50_jaundice_model.h5")
