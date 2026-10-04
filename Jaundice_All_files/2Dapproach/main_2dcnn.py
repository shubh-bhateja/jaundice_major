"""
main_2dcnn.py
A self-contained script for multimodal neonatal jaundice prediction using a 2D CNN for images and MLP for clinical data.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate)
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# --- CONFIG ---
IMG_SIZE = (128, 128)
IMG_DIR = '../images'  # Adjust if needed
CLINICAL_CSV = '../NeoJaundice/chd_jaundice_published_2.csv'  # Adjust if needed
BATCH_SIZE = 16
EPOCHS = 10

# --- 1. Preprocessing functions ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    # Example: simple center crop (replace with ROI extraction/color calibration as needed)
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img = img[start_h:start_h+min_dim, start_w:start_w+min_dim]
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

# --- 2. Data loading ---
def load_clinical_data():
    df = pd.read_csv(CLINICAL_CSV)
    # Map columns as needed (example: patient_id, blood(mg/dL), Treatment)
    df = df.rename(columns={
        'patient_id': 'Infant_ID',
        'blood(mg/dL)': 'TSB',
        'Treatment': 'Risk_Class'
    })
    return df

def get_image_path(infant_id):
    # Example: expects images named like 0001-1.jpg, 0001-2.jpg, etc.
    for suffix in ['-1.jpg', '-2.jpg', '-3.jpg']:
        path = os.path.join(IMG_DIR, f"{str(infant_id).zfill(4)}{suffix}")
        if os.path.exists(path):
            return path
    return None

# --- 3. Data generator ---
class DataGen(Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = self.df.iloc[batch_idx]
        X_img = []
        X_clin = []
        y_reg = []
        y_cls = []
        for _, row in batch.iterrows():
            img_path = get_image_path(row['Infant_ID'])
            if img_path is None:
                continue
            img = preprocess_image(img_path)
            X_img.append(img)
            # Example: use 3 clinical features (customize as needed)
            X_clin.append([row['TSB'], row.get('feature2', 0), row.get('feature3', 0)])
            y_reg.append(row['TSB'])
            # Example: risk class as integer (0,1,2)
            y_cls.append(int(row['Risk_Class']))
        return [np.array(X_img), np.array(X_clin)], [np.array(y_reg), np.array(y_cls)]
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- 4. Model definition ---
def build_model():
    # Visual branch (2D CNN)
    visual_input = Input(shape=(*IMG_SIZE, 3), name='visual_input')
    x = Conv2D(32, (3,3), activation='relu', padding='same')(visual_input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    visual_embedding = Dense(64, activation='relu')(x)
    # Clinical branch (MLP)
    clinical_input = Input(shape=(3,), name='clinical_input')
    c = Dense(32, activation='relu')(clinical_input)
    c = Dropout(0.3)(c)
    clinical_embedding = Dense(16, activation='relu')(c)
    # Fusion
    fusion = Concatenate()([visual_embedding, clinical_embedding])
    f = Dense(64, activation='relu')(fusion)
    f = Dropout(0.3)(f)
    # Outputs
    tsb_regression = Dense(1, name='tsb_regression_output')(f)
    risk_classification = Dense(3, activation='softmax', name='risk_classification_output')(f)
    model = Model(inputs=[visual_input, clinical_input], outputs=[tsb_regression, risk_classification])
    model.compile(
        optimizer='adam',
        loss={'tsb_regression_output': 'mse', 'risk_classification_output': 'sparse_categorical_crossentropy'},
        metrics={'tsb_regression_output': 'mae', 'risk_classification_output': 'accuracy'}
    )
    return model

# --- 5. Main script ---
if __name__ == '__main__':
    df = load_clinical_data()
    # Example: drop rows without images
    df['img_path'] = df['Infant_ID'].apply(get_image_path)
    df = df[df['img_path'].notnull()]
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_gen = DataGen(train_df)
    val_gen = DataGen(val_df, shuffle=False)
    # Build model
    model = build_model()
    model.summary()
    # Train
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    # Save
    model.save('2dcnn_multimodal_model.h5')
    print('Training complete. Model saved as 2dcnn_multimodal_model.h5')
