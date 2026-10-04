"""
Neonatal Jaundice Detection using 1D Convolutional Neural Network
=================================================================
Replication of: "Artificial intelligence-based noninvasive bilirubin prediction
for neonatal jaundice using 1D convolutional neural network" (Makhloughi, 2025)

Dataset: NeoJaundice dataset from Xuzhou Central Hospital
         https://springernature.figshare.com/articles/dataset/NeoJaundice_Neonatal_Jaundice_Evaluation_in_Demographic_Images/22302559

Usage:
    python neonatal_jaundice_1dcnn.py

Adjust the DATA_DIR and CSV_PATH constants below to match your local paths.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, mean_squared_error,
                             mean_absolute_error, r2_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit these paths
# ─────────────────────────────────────────────
DATA_DIR  = "/Users/shubhbhateja/College/minor_project/images"          # folder that contains the images
CSV_PATH  = "/Users/shubhbhateja/College/minor_project/NeoJaundice/chd_jaundice_published_2.csv"  # CSV with columns: filename, tsb
                                        # (adjust column names below if different)
IMG_COL   = "filename"               # column name for image filenames
TSB_COL   = "TSB"                    # column name for Total Serum Bilirubin (mg/dL)

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model / training constants (from paper)
IMG_SIZE   = 567        # paper states average 567×567; we crop a fixed ROI
ROI_SIZE   = 100        # fixed-size skin ROI (pixels square)
N_PIXELS   = ROI_SIZE * ROI_SIZE   # = 10,000 time-series length per channel
BATCH_SIZE = 32
EPOCHS     = 150
LR_INIT    = 0.01
SEED       = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
#  1. PREPROCESSING
# ─────────────────────────────────────────────

def white_balance_with_color_card(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the yellow reference card in the image using HSV thresholding,
    compute its average LAB values, and apply a per-channel shift to the
    whole image so that the card looks 'standard'. Returns BGR image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Yellow hue range in OpenCV (H: 0-179)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr  # no card found → return as-is

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    card_region = img_bgr[y:y+h, x:x+w]

    # Reference: ideal LAB for a standard yellow card (approximate)
    ref_lab = np.array([75.0, 5.0, 60.0])

    card_lab   = cv2.cvtColor(card_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab   = card_lab.mean(axis=(0, 1))
    correction = ref_lab - mean_lab

    img_lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab   += correction
    img_lab    = np.clip(img_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


def extract_skin_roi(img_bgr: np.ndarray, roi_size: int = ROI_SIZE) -> np.ndarray:
    """
    Extract a fixed center ROI (skin area) from the image.
    The paper uses a fixed coordinate window; we use the image center.
    """
    h, w = img_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    half   = roi_size // 2
    roi    = img_bgr[cy-half:cy+half, cx-half:cx+half]
    if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
        roi = cv2.resize(roi, (roi_size, roi_size))
    return roi


def extract_color_channels(roi_bgr: np.ndarray) -> dict:
    """
    Convert ROI to RGB, HSV, LAB, YCbCr and flatten each channel to 1-D.
    Returns a dict: {space_channel_name: np.ndarray of shape (N_PIXELS,)}
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    roi_hsv[:,:,0] /= 179.0   # H → [0,1]
    roi_hsv[:,:,1:] /= 255.0  # S,V → [0,1]
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi_lab[:,:,0] /= 100.0   # L → [0,1]
    roi_lab[:,:,1:] = (roi_lab[:,:,1:] + 128) / 255.0  # a,b → [0,1]
    roi_ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0

    channels = {}
    for ch_i, ch_name in enumerate(['R','G','B']):
        channels[ch_name] = roi_rgb[:,:,ch_i].flatten()
    for ch_i, ch_name in enumerate(['H','S','V']):
        channels[ch_name] = roi_hsv[:,:,ch_i].flatten()
    for ch_i, ch_name in enumerate(['L','A','B_lab']):
        channels[ch_name] = roi_lab[:,:,ch_i].flatten()
    for ch_i, ch_name in enumerate(['Y','Cb','Cr']):
        channels[ch_name] = roi_ycbcr[:,:,ch_i].flatten()
    return channels


def preprocess_image(img_path: str) -> dict | None:
    """Full preprocessing pipeline for one image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = white_balance_with_color_card(img)
    roi = extract_skin_roi(img)
    return extract_color_channels(roi)


# ─────────────────────────────────────────────
#  2. DATASET LOADING
# ─────────────────────────────────────────────

COLOR_SPACES = {
    "RGB":    ["R", "G", "B"],
    "HSV":    ["H", "S", "V"],
    "LAB":    ["L", "A", "B_lab"],
    "YCbCr":  ["Y", "Cb", "Cr"],
}

def build_feature_array(channels: dict, space: str) -> np.ndarray:
    """Stack the 1-D channels for a given color space → (N_PIXELS * n_ch,)"""
    keys = COLOR_SPACES[space]
    return np.concatenate([channels[k] for k in keys])


def load_dataset(data_dir: str, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        all_features : dict of space → np.ndarray (N_samples, N_PIXELS*3)
        labels_tsb   : np.ndarray (N_samples,)  continuous TSB values
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows")

    all_channels = {k: [] for space in COLOR_SPACES for k in COLOR_SPACES[space]}
    tsb_values   = []
    skipped      = 0

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, str(row[IMG_COL]))
        if not os.path.exists(img_path):
            skipped += 1
            continue
        channels = preprocess_image(img_path)
        if channels is None:
            skipped += 1
            continue
        for k, v in channels.items():
            all_channels[k].append(v)
        tsb_values.append(float(row[TSB_COL]))

    print(f"Processed {len(tsb_values)} images, skipped {skipped}")

    # Build per-space feature matrices
    features = {}
    for space, keys in COLOR_SPACES.items():
        mat = np.stack([np.concatenate([all_channels[k][i] for k in keys])
                        for i in range(len(tsb_values))], axis=0)
        features[space] = mat.astype(np.float32)

    labels = np.array(tsb_values, dtype=np.float32)
    return features, labels


def tsb_to_class(tsb: np.ndarray) -> np.ndarray:
    """Map continuous TSB (mg/dL) to 3-class labels: 0→[0-5], 1→[5-10], 2→[10+]"""
    cls = np.zeros(len(tsb), dtype=int)
    cls[(tsb >= 5) & (tsb < 10)] = 1
    cls[tsb >= 10] = 2
    return cls


# ─────────────────────────────────────────────
#  3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_1dcnn_branch(input_length: int, name_prefix: str = "") -> tuple:
    """
    Single 1DCNN branch as described in Table 2 of the paper.
    Returns (input_tensor, feature_tensor).
    """
    inp = keras.Input(shape=(input_length, 1), name=f"{name_prefix}input")
    x = layers.Conv1D(32,  3, padding="same", activation="relu", name=f"{name_prefix}conv1")(inp)
    x = layers.MaxPooling1D(2, padding="same", name=f"{name_prefix}pool1")(x)
    x = layers.Conv1D(64,  3, padding="same", activation="relu", name=f"{name_prefix}conv2")(x)
    x = layers.MaxPooling1D(2, padding="same", name=f"{name_prefix}pool2")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name=f"{name_prefix}conv3")(x)
    x = layers.MaxPooling1D(2, padding="same", name=f"{name_prefix}pool3")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name=f"{name_prefix}conv4")(x)
    x = layers.MaxPooling1D(2, padding="same", name=f"{name_prefix}pool4")(x)
    x = layers.Flatten(name=f"{name_prefix}flatten")(x)
    return inp, x


def build_regression_model(input_length: int, space_combo: list[str]) -> keras.Model:
    """
    Build a single- or dual-branch 1DCNN regression model.
    space_combo : list of 1 or 2 color-space names, e.g. ["RGB"] or ["RGB","HSV"]
    """
    branches_in, branches_out = [], []
    for sp in space_combo:
        n_ch = len(COLOR_SPACES[sp])
        inp, feat = build_1dcnn_branch(input_length * n_ch, name_prefix=f"{sp}_")
        branches_in.append(inp)
        branches_out.append(feat)

    merged = layers.concatenate(branches_out) if len(branches_out) > 1 else branches_out[0]

    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="linear", name="tsb_output")(x)

    model = keras.Model(inputs=branches_in, outputs=out)
    return model


def build_classification_model(input_length: int, space_combo: list[str],
                                n_classes: int = 3) -> keras.Model:
    """Same architecture but with softmax output for 3-class classification."""
    branches_in, branches_out = [], []
    for sp in space_combo:
        n_ch = len(COLOR_SPACES[sp])
        inp, feat = build_1dcnn_branch(input_length * n_ch, name_prefix=f"{sp}_")
        branches_in.append(inp)
        branches_out.append(feat)

    merged = layers.concatenate(branches_out) if len(branches_out) > 1 else branches_out[0]

    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_output")(x)

    model = keras.Model(inputs=branches_in, outputs=out)
    return model


# ─────────────────────────────────────────────
#  4. TRAINING UTILITIES
# ─────────────────────────────────────────────

def get_callbacks() -> list:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6, verbose=0),
    ]


def prepare_inputs(X_dict: dict, space_combo: list[str]) -> list[np.ndarray]:
    """Return a list of arrays (one per branch) shaped (N, L, 1)."""
    inputs = []
    for sp in space_combo:
        arr = X_dict[sp]  # (N, L)
        inputs.append(arr[..., np.newaxis])   # (N, L, 1)
    return inputs


def train_regression(features: dict, labels: np.ndarray,
                     space_combo: list[str], n_runs: int = 10) -> dict:
    """
    Train regression model n_runs times and average metrics (paper trains 10×).
    Returns dict with mean RMSE, MAE, R2.
    """
    combo_name = "+".join(space_combo)
    print(f"\n{'='*60}")
    print(f"  Regression  |  {combo_name}")
    print(f"{'='*60}")

    rmse_list, mae_list, r2_list = [], [], []

    for run in range(n_runs):
        # 70/10/20 split
        idx = np.arange(len(labels))
        idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=run)
        idx_val, idx_te = train_test_split(idx_tmp, test_size=0.667, random_state=run)

        def split(arr, idx_): return arr[idx_]

        X_tr  = {sp: split(features[sp], idx_tr)  for sp in space_combo}
        X_val = {sp: split(features[sp], idx_val) for sp in space_combo}
        X_te  = {sp: split(features[sp], idx_te)  for sp in space_combo}
        y_tr, y_val, y_te = labels[idx_tr], labels[idx_val], labels[idx_te]

        inp_len = features[space_combo[0]].shape[1] // len(COLOR_SPACES[space_combo[0]])
        model   = build_regression_model(inp_len, space_combo)
        opt     = keras.optimizers.Adam(learning_rate=LR_INIT)
        model.compile(optimizer=opt, loss="mse")

        model.fit(
            prepare_inputs(X_tr, space_combo), y_tr,
            validation_data=(prepare_inputs(X_val, space_combo), y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=get_callbacks(), verbose=0,
        )

        y_pred = model.predict(prepare_inputs(X_te, space_combo), verbose=0).flatten()
        rmse_list.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae_list.append(mean_absolute_error(y_te, y_pred))
        r2_list.append(r2_score(y_te, y_pred))
        print(f"  Run {run+1:2d} | RMSE={rmse_list[-1]:.3f}  MAE={mae_list[-1]:.3f}  R²={r2_list[-1]:.3f}")

    result = {
        "combo":  combo_name,
        "RMSE":   np.mean(rmse_list),
        "MAE":    np.mean(mae_list),
        "R2":     np.mean(r2_list),
    }
    print(f"\n  AVG | RMSE={result['RMSE']:.3f}  MAE={result['MAE']:.3f}  R²={result['R2']:.3f}")
    return result


def train_classification(features: dict, labels_cls: np.ndarray,
                          space_combo: list[str] = ["RGB", "HSV"]) -> dict:
    """
    10-fold cross-validation classification with the best color-space combo (RGB+HSV).
    Returns dict with mean Accuracy, Precision, Recall, F1.
    """
    combo_name = "+".join(space_combo)
    print(f"\n{'='*60}")
    print(f"  Classification  |  {combo_name}  (10-fold CV)")
    print(f"{'='*60}")

    n_classes  = 3
    kf         = KFold(n_splits=10, shuffle=True, random_state=SEED)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    all_preds, all_true = [], []

    combo_features = np.concatenate([features[sp] for sp in space_combo], axis=1)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(combo_features)):
        X_tr_raw, X_te_raw = combo_features[tr_idx], combo_features[te_idx]
        y_tr_raw, y_te_raw = labels_cls[tr_idx], labels_cls[te_idx]

        # Further split train into train/val
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr_raw, y_tr_raw, test_size=0.1, random_state=fold, stratify=y_tr_raw)

        # Rebuild per-space dicts for this fold
        lens = [features[sp].shape[1] for sp in space_combo]
        cum  = np.cumsum([0] + lens)

        def make_dict(arr):
            return {sp: arr[:, cum[i]:cum[i+1]] for i, sp in enumerate(space_combo)}

        Xd_tr  = make_dict(X_tr2)
        Xd_val = make_dict(X_val)
        Xd_te  = make_dict(X_te_raw)

        inp_len = features[space_combo[0]].shape[1] // len(COLOR_SPACES[space_combo[0]])
        model   = build_classification_model(inp_len, space_combo, n_classes)
        opt     = keras.optimizers.Adam(learning_rate=LR_INIT)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        model.fit(
            prepare_inputs(Xd_tr, space_combo), y_tr2,
            validation_data=(prepare_inputs(Xd_val, space_combo), y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=get_callbacks(), verbose=0,
        )

        y_pred = np.argmax(
            model.predict(prepare_inputs(Xd_te, space_combo), verbose=0), axis=1)
        all_preds.extend(y_pred.tolist())
        all_true.extend(y_te_raw.tolist())

        acc  = accuracy_score(y_te_raw, y_pred) * 100
        prec = precision_score(y_te_raw, y_pred, average="weighted", zero_division=0) * 100
        rec  = recall_score(y_te_raw, y_pred, average="weighted", zero_division=0) * 100
        f1   = f1_score(y_te_raw, y_pred, average="weighted", zero_division=0) * 100
        acc_list.append(acc); prec_list.append(prec)
        rec_list.append(rec); f1_list.append(f1)
        print(f"  Fold {fold+1:2d} | Acc={acc:.2f}%  Prec={prec:.2f}%  Rec={rec:.2f}%  F1={f1:.2f}%")

    result = {
        "combo":     combo_name,
        "Accuracy":  np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall":    np.mean(rec_list),
        "F1":        np.mean(f1_list),
    }
    print(f"\n  AVG | Acc={result['Accuracy']:.2f}%  Prec={result['Precision']:.2f}%"
          f"  Rec={result['Recall']:.2f}%  F1={result['F1']:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plot_confusion_matrix(cm, ["0-5", "5-10", "10+"], combo_name)
    return result


# ─────────────────────────────────────────────
#  5. VISUALISATION
# ─────────────────────────────────────────────

def plot_channel_comparison(features: dict, labels: np.ndarray,
                            n_pixels: int = 500):
    """Reproduce Fig. 4 – pixel-level channel plots for normal vs jaundiced baby."""
    normal_idx  = np.where(labels < 10)[0][0]
    jaundice_idx = np.where(labels >= 15)[0][0]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("Color Channel Comparison: Normal vs Jaundiced Baby", fontsize=14)

    space_channel_map = {
        "RGB":   (["R","G","B"],   ["red","green","blue"]),
        "HSV":   (["H","S","V"],   ["purple","orange","cyan"]),
        "LAB":   (["L","A","B_lab"], ["gray","magenta","gold"]),
        "YCbCr": (["Y","Cb","Cr"],  ["gold","blue","red"]),
    }

    for row, (space, (keys, colors)) in enumerate(space_channel_map.items()):
        for col, idx in enumerate([normal_idx, jaundice_idx]):
            ax = axes[row][col]
            full = features[space][idx]      # (L,)
            ch_len = len(full) // len(keys)
            for ki, (key, color) in enumerate(zip(keys, colors)):
                ch = full[ki*ch_len:(ki+1)*ch_len][:n_pixels]
                ax.plot(ch, color=color, label=key, linewidth=0.8)
            title = f"{space} – {'Normal' if col==0 else 'Jaundiced'}"
            title += f"\nTSB = {labels[idx]:.1f} mg/dL"
            ax.set_title(title)
            ax.legend(); ax.set_xlabel("Pixel Index"); ax.set_ylabel("Intensity")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "channel_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_regression_results(results: list[dict]):
    """Bar chart of regression metrics across color space combos."""
    df = pd.DataFrame(results).set_index("combo")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        df[metric].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Regression – {metric}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "regression_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_classification_results(results: list[dict]):
    """Bar chart of classification metrics across color space combos."""
    df = pd.DataFrame(results).set_index("combo")
    ax = df[["Accuracy","Precision","Recall","F1"]].plot(
        kind="bar", figsize=(14, 6), edgecolor="black")
    ax.set_title("Classification Metrics by Color Space")
    ax.set_ylabel("Score (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "classification_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, combo: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix – {combo}")
    plt.tight_layout()
    safe = combo.replace("+", "_")
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
#  6. MAIN PIPELINE
# ─────────────────────────────────────────────

REGRESSION_COMBOS = [
    ["RGB"], ["HSV"], ["LAB"], ["YCbCr"],
    ["RGB", "HSV"], ["RGB", "LAB"], ["RGB", "YCbCr"],
    ["HSV", "LAB"], ["HSV", "YCbCr"], ["LAB", "YCbCr"],
]

CLASSIFICATION_COMBOS = [
    ["RGB"], ["HSV"], ["LAB"], ["YCbCr"],
    ["RGB", "HSV"], ["RGB", "LAB"], ["RGB", "YCbCr"],
    ["HSV", "LAB"], ["HSV", "YCbCr"], ["LAB", "YCbCr"],
]


def main():
    # ── Load dataset ──────────────────────────────
    print("Loading and preprocessing dataset …")
    features, labels_tsb = load_dataset(DATA_DIR, CSV_PATH)
    labels_cls = tsb_to_class(labels_tsb)

    print(f"\nDataset stats:")
    print(f"  Total samples : {len(labels_tsb)}")
    print(f"  TSB range     : {labels_tsb.min():.1f} – {labels_tsb.max():.1f} mg/dL")
    print(f"  Class 0 (0-5)   : {(labels_cls==0).sum()}")
    print(f"  Class 1 (5-10)  : {(labels_cls==1).sum()}")
    print(f"  Class 2 (10+)   : {(labels_cls==2).sum()}")

    # ── Channel comparison plot (Fig. 4) ──────────
    plot_channel_comparison(features, labels_tsb)

    # ── Regression experiments ────────────────────
    reg_results = []
    for combo in REGRESSION_COMBOS:
        res = train_regression(features, labels_tsb, combo, n_runs=10)
        reg_results.append(res)

    reg_df = pd.DataFrame(reg_results)
    print("\n\n=== REGRESSION SUMMARY ===")
    print(reg_df.to_string(index=False))
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)
    plot_regression_results(reg_results)

    # ── Classification experiments ────────────────
    cls_results = []
    for combo in CLASSIFICATION_COMBOS:
        res = train_classification(features, labels_cls, combo)
        cls_results.append(res)

    cls_df = pd.DataFrame(cls_results)
    print("\n\n=== CLASSIFICATION SUMMARY ===")
    print(cls_df.to_string(index=False))
    cls_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)
    plot_classification_results(cls_results)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()