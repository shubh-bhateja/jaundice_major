#!/usr/bin/env python3
"""
NeoJaundice Training Pipeline.

This script combines the high-performance 2D-CNN/Transfer-Learning experiments
with a convenient default configuration for the best-performing model.

Features:
  - RGB+HSV six-channel image tensors.
  - Optional clinical metadata fusion (gender, gestational age, age, weight).
  - MobileNetV2 transfer learning (default) and compact 2D-CNN backbones.
  - Support for image-level and patient-level splits.
  - Automated skin cropping and calibration (if needed).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes script is in hpc_files/
DEFAULT_DATA_DIR = PROJECT_ROOT / "Jaundice_All_files" / "NeoJaundice"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "chd_jaundice_published_2.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Jaundice_All_files" / "original_cropped_skin_patches"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train_outputs"

CLASS_NAMES = ("0-5", "5-10", ">10")
NUM_CLASSES = 3
META_COLUMNS = ("gender", "gestational_age", "age(day)", "weight")

YELLOW_CARD_REFERENCE_CIELAB = (75.0, 5.0, 60.0)
YELLOW_CARD_REFERENCE_OPENCV = (
    YELLOW_CARD_REFERENCE_CIELAB[0] * 255.0 / 100.0,
    YELLOW_CARD_REFERENCE_CIELAB[1] + 128.0,
    YELLOW_CARD_REFERENCE_CIELAB[2] + 128.0,
)


@dataclass(frozen=True)
class Sample:
    patient_id: str
    image_idx: str
    image_path: Path
    bilirubin: float
    class_id: int
    metadata: tuple[float, ...]


def validate_paths(args: argparse.Namespace) -> None:
    """Ensure required dataset files exist before starting."""
    csv_path = resolve_csv_path(args)
    image_dir = args.image_dir
    
    missing = []
    if not csv_path.exists():
        missing.append(str(csv_path))
    if not image_dir.exists():
        missing.append(str(image_dir))

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Required dataset files/folders are missing:\n"
            f"  {joined}\n\n"
            "Make sure you are running this script from the folder containing "
            "the 'NeoJaundice' data and 'original_cropped_skin_patches'."
        )


def import_dependencies():
    missing = []
    try:
        import cv2  # noqa: F401
    except Exception:
        missing.append("opencv-python")
    try:
        import numpy as np  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        missing.append("pandas")
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        missing.append("tensorflow")
    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")

    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall them with:\n  python -m pip install "
            + " ".join(missing)
        )

    import cv2
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception:
        StratifiedGroupKFold = None

    return {
        "cv2": cv2,
        "np": np,
        "pd": pd,
        "tf": tf,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "precision_score": precision_score,
        "r2_score": r2_score,
        "recall_score": recall_score,
        "StratifiedKFold": StratifiedKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "StratifiedShuffleSplit": StratifiedShuffleSplit,
    }


def parse_args() -> argparse.Namespace:
    # Use best-tested configuration as defaults
    default_image_dir = DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.exists() else DEFAULT_DATA_DIR / "images"
    
    parser = argparse.ArgumentParser(
        description="NeoJaundice Training: MobileNetV2 + HSV + Metadata Fusion."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("compact_2d", "mobilenetv2"),
        default="mobilenetv2",
        help="mobilenetv2 is the recommended high-performance backbone.",
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "multitask"),
        default="classification",
        help="Task type: 3-class classification, bilirubin regression, or both.",
    )
    parser.add_argument(
        "--split",
        choices=("patient-kfold", "image-kfold", "patient-holdout", "image-holdout"),
        default="image-holdout",
        help="Use image-holdout for quick 80/20 verification or patient-* for robust evaluation.",
    )
    parser.add_argument(
        "--calibration",
        choices=("auto", "none", "gray-world", "yellow-card"),
        default="auto",
        help="Color calibration method.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick verification pass with tiny data.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int, tf_module) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    tf_module.random.set_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def configure_accelerator(tf_module) -> None:
    gpus = tf_module.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf_module.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if gpus:
        print(f"Detected TensorFlow GPU(s): {[gpu.name for gpu in gpus]}")
        return

    try:
        cpus = tf_module.config.list_physical_devices("CPU")
        print(f"No TensorFlow GPU detected; using CPU devices: {[cpu.name for cpu in cpus]}")
    except Exception:
        print("No TensorFlow GPU detected; using CPU.")


def bilirubin_to_class(value: float) -> int:
    if value < 5.0: return 0
    if value < 10.0: return 1
    return 2


def encode_gender(value: str) -> float:
    text = str(value).strip().upper()
    if text == "M": return 1.0
    if text == "F": return 0.0
    return 0.5


def safe_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metadata_from_row(row: dict[str, str]) -> tuple[float, ...]:
    return (
        encode_gender(row.get("gender", "")),
        safe_float(row.get("gestational_age", float("nan"))),
        safe_float(row.get("age(day)", float("nan"))),
        safe_float(row.get("weight", float("nan"))),
    )


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv_path is not None:
        return args.csv_path
    return args.data_dir / "chd_jaundice_published_2.csv"


def load_samples(args: argparse.Namespace, np_module) -> list[Sample]:
    csv_path = resolve_csv_path(args)
    image_dir = args.image_dir
    
    samples: list[Sample] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"patient_id", "image_idx", "blood(mg/dL)"}
        missing = required.difference(fieldnames)
        if missing:
            raise KeyError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            image_idx = row["image_idx"]
            image_path = image_dir / image_idx
            if not image_path.exists():
                continue
            bilirubin = float(row["blood(mg/dL)"])
            samples.append(
                Sample(
                    patient_id=str(row["patient_id"]),
                    image_idx=image_idx,
                    image_path=image_path,
                    bilirubin=bilirubin,
                    class_id=bilirubin_to_class(bilirubin),
                    metadata=metadata_from_row(row),
                )
            )

    if args.smoke:
        by_class = {0: [], 1: [], 2: []}
        for sample in samples:
            if len(by_class[sample.class_id]) < 36:
                by_class[sample.class_id].append(sample)
        samples = by_class[0] + by_class[1] + by_class[2]

    if not samples:
        raise ValueError(f"No image rows could be loaded from {image_dir}")

    y = np_module.array([sample.class_id for sample in samples], dtype=np_module.int64)
    unique, counts = np_module.unique(y, return_counts=True)
    print(f"Loaded {len(samples)} images from {image_dir}")
    print(f"Patients: {len(set(sample.patient_id for sample in samples))}")
    print("Class counts:")
    for class_id, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[int(class_id)]:>4}: {int(count)}")
    print(f"Majority-class baseline accuracy: {counts.max() / counts.sum() * 100:.2f}%")
    return samples


def yellow_card_mask(bgr, cv2_module, np_module):
    hsv = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2HSV)
    lower_yellow = np_module.array([15, 80, 80], dtype=np_module.uint8)
    upper_yellow = np_module.array([35, 255, 255], dtype=np_module.uint8)
    mask = cv2_module.inRange(hsv, lower_yellow, upper_yellow)
    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (5, 5))
    return cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel, iterations=2)


def yellow_card_white_balance(bgr, cv2_module, np_module):
    mask = yellow_card_mask(bgr, cv2_module, np_module)
    contours, _ = cv2_module.findContours(mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    if not contours: return bgr

    largest = max(contours, key=cv2_module.contourArea)
    if cv2_module.contourArea(largest) < 200: return bgr

    card_mask = np_module.zeros(mask.shape, dtype=np_module.uint8)
    cv2_module.drawContours(card_mask, [largest], contourIdx=-1, color=255, thickness=-1)

    lab = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2LAB).astype(np_module.float32)
    mean_lab = cv2_module.mean(lab, mask=card_mask)[:3]
    correction = np_module.array(YELLOW_CARD_REFERENCE_OPENCV, dtype=np_module.float32) - np_module.array(mean_lab, dtype=np_module.float32)
    balanced = lab + correction.reshape(1, 1, 3)
    balanced = np_module.clip(balanced, 0, 255).astype(np_module.uint8)
    return cv2_module.cvtColor(balanced, cv2_module.COLOR_LAB2BGR)


def gray_world_lab_balance(bgr, cv2_module, np_module):
    lab = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2LAB).astype(np_module.float32)
    avg_a = np_module.mean(lab[:, :, 1])
    avg_b = np_module.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 1.1
    lab = np_module.clip(lab, 0, 255).astype(np_module.uint8)
    return cv2_module.cvtColor(lab, cv2_module.COLOR_LAB2BGR)


def adaptive_skin_crop_from_full_image(bgr, cv2_module, np_module):
    height, width = bgr.shape[:2]
    ycrcb = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2YCrCb)
    mask = cv2_module.inRange(
        ycrcb,
        np_module.array([0, 133, 77], dtype=np_module.uint8),
        np_module.array([255, 173, 127], dtype=np_module.uint8),
    )
    mask[int(height * 0.82) :, :] = 0
    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_ELLIPSE, (7, 7))
    mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_OPEN, kernel, iterations=2)
    mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2_module.findContours(mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    if not contours:
        size = min(height, width)
        y1, x1 = (height - size) // 2, (width - size) // 2
        return bgr[y1 : y1 + size, x1 : x1 + size]

    contour = max(contours, key=cv2_module.contourArea)
    x, y, w, h = cv2_module.boundingRect(contour)
    ix, iy = int(w * 0.05), int(h * 0.05)
    x1, y1 = max(0, x + ix), max(0, y + iy)
    x2, y2 = min(width, x + w - ix), min(height, y + h - iy)
    return bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else bgr


def choose_calibration(mode: str, image_dir: Path, bgr_shape: tuple[int, int, int]) -> str:
    if mode != "auto": return mode
    if image_dir.name == "original_cropped_skin_patches": return "none"
    if bgr_shape[0] >= 500 and bgr_shape[1] >= 500: return "yellow-card"
    return "none"


def preprocess_image(sample: Sample, args: argparse.Namespace, deps):
    cv2_module, np_module = deps["cv2"], deps["np"]
    bgr = cv2_module.imread(str(sample.image_path), cv2_module.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"Could not read: {sample.image_path}")

    cal = choose_calibration(args.calibration, args.image_dir, bgr.shape)
    if cal == "yellow-card": bgr = yellow_card_white_balance(bgr, cv2_module, np_module)
    elif cal == "gray-world": bgr = gray_world_lab_balance(bgr, cv2_module, np_module)

    if bgr.shape[0] >= 500 and bgr.shape[1] >= 500 and args.image_dir.name != "original_cropped_skin_patches":
        bgr = adaptive_skin_crop_from_full_image(bgr, cv2_module, np_module)

    bgr = cv2_module.resize(bgr, (args.image_size, args.image_size), interpolation=cv2_module.INTER_AREA)
    rgb = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    hsv = cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2HSV).copy()
    hsv[:, :, 0] = np_module.round(hsv[:, :, 0].astype(np_module.float32) * 255.0 / 179.0).clip(0, 255)
    return np_module.concatenate([rgb, hsv], axis=-1).astype(np_module.uint8)


def cache_file(args: argparse.Namespace) -> Path:
    name = f"rgb_hsv_{args.image_size}_{args.image_dir.name}_{args.calibration}.npy"
    return args.output_dir / "cache" / name


def labels_cache_file(args: argparse.Namespace) -> Path:
    name = f"labels_{args.image_dir.name}.json"
    return args.output_dir / "cache" / name


def load_or_build_cache(samples: list[Sample], args: argparse.Namespace, deps):
    np_module = deps["np"]
    (args.output_dir / "cache").mkdir(parents=True, exist_ok=True)
    x_path, labels_path = cache_file(args), labels_cache_file(args)

    image_names = [sample.image_idx for sample in samples]
    if not args.rebuild_cache and x_path.exists() and labels_path.exists():
        with labels_path.open() as h: payload = json.load(h)
        if payload.get("image_idx") == image_names and payload.get("image_size") == args.image_size:
            print(f"Loading image cache: {x_path}")
            return np_module.load(x_path, mmap_mode=None)

    print("Building 2D RGB+HSV image cache...")
    start = time.time()
    X = np_module.zeros((len(samples), args.image_size, args.image_size, 6), dtype=np_module.uint8)
    for i, sample in enumerate(samples):
        X[i] = preprocess_image(sample, args, deps)
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(samples):
            print(f"  processed {i + 1:4d}/{len(samples)} images ({time.time() - start:.1f}s)")

    np_module.save(x_path, X)
    with labels_path.open("w") as h:
        json.dump({
            "image_idx": image_names,
            "patient_id": [sample.patient_id for sample in samples],
            "bilirubin": [sample.bilirubin for sample in samples],
            "class_id": [sample.class_id for sample in samples],
            "image_size": args.image_size,
        }, h, indent=2)
    return X


def raw_arrays(samples: list[Sample], np_module):
    y_cls = np_module.array([sample.class_id for sample in samples], dtype=np_module.int64)
    y_reg = np_module.array([sample.bilirubin for sample in samples], dtype=np_module.float32)
    groups = np_module.array([sample.patient_id for sample in samples])
    metadata = np_module.array([sample.metadata for sample in samples], dtype=np_module.float32)
    return y_cls, y_reg, groups, metadata


def make_folds(y_cls, groups, args: argparse.Namespace, deps):
    np_module = deps["np"]
    SKF, SGKF, SSS = deps["StratifiedKFold"], deps["StratifiedGroupKFold"], deps["StratifiedShuffleSplit"]

    n_splits = 2 if args.smoke else args.folds
    if args.split == "image-kfold":
        return list(SKF(n_splits=n_splits, shuffle=True, random_state=args.seed).split(np_module.zeros(len(y_cls)), y_cls))
    if args.split == "patient-kfold":
        if SGKF is None: raise RuntimeError("StratifiedGroupKFold not available.")
        return list(SGKF(n_splits=n_splits, shuffle=True, random_state=args.seed).split(np_module.zeros(len(y_cls)), y_cls, groups))
    if args.split == "image-holdout":
        return list(SSS(n_splits=1, test_size=0.2, random_state=args.seed).split(np_module.zeros(len(y_cls)), y_cls))
    if args.split == "patient-holdout":
        if SGKF is None: raise RuntimeError("StratifiedGroupKFold not available.")
        return [next(SGKF(n_splits=5, shuffle=True, random_state=args.seed).split(np_module.zeros(len(y_cls)), y_cls, groups))]
    raise ValueError(f"Unknown split: {args.split}")


def validation_split(train_idx, y_cls, groups, args: argparse.Namespace, deps):
    np_module = deps["np"]
    SGKF, SSS = deps["StratifiedGroupKFold"], deps["StratifiedShuffleSplit"]
    if args.split.startswith("patient"):
        if SGKF is None: raise RuntimeError("StratifiedGroupKFold not available.")
        tr, val = next(SGKF(n_splits=5, shuffle=True, random_state=args.seed+101).split(np_module.zeros(len(train_idx)), y_cls[train_idx], groups[train_idx]))
        return train_idx[tr], train_idx[val]
    tr, val = next(SSS(n_splits=1, test_size=0.125, random_state=args.seed+101).split(np_module.zeros(len(train_idx)), y_cls[train_idx]))
    return train_idx[tr], train_idx[val]


def metadata_transform(metadata, train_idx, np_module):
    train_meta = metadata[train_idx].astype(np_module.float32)
    medians = np_module.nanmedian(train_meta, axis=0)
    medians = np_module.where(np_module.isfinite(medians), medians, 0.0)
    filled = np_module.where(np_module.isnan(metadata), medians.reshape(1, -1), metadata)
    train_filled = filled[train_idx]
    mean, std = train_filled.mean(axis=0), train_filled.std(axis=0)
    std = np_module.where(std < 1e-6, 1.0, std)
    return ((filled - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np_module.float32)


def regression_transform(y_reg, train_idx, np_module):
    mean, std = float(y_reg[train_idx].mean()), float(y_reg[train_idx].std())
    if std < 1e-6: std = 1.0
    return ((y_reg - mean) / std).astype(np_module.float32), mean, std


def class_alpha(y_train, np_module):
    counts = np_module.bincount(y_train, minlength=NUM_CLASSES).astype(np_module.float32)
    weights = counts.sum() / (NUM_CLASSES * np_module.maximum(counts, 1.0))
    return (weights / weights.mean()).astype(np_module.float32)


def sparse_focal_loss(alpha_values, gamma: float, tf_module):
    alpha = tf_module.constant(alpha_values, dtype=tf_module.float32)
    def loss(y_true, y_pred):
        y_true_i = tf_module.cast(tf_module.reshape(y_true, [-1]), tf_module.int32)
        y_pred_c = tf_module.clip_by_value(y_pred, tf_module.keras.backend.epsilon(), 1.0 - tf_module.keras.backend.epsilon())
        p_t = tf_module.reduce_sum(tf_module.one_hot(y_true_i, depth=NUM_CLASSES) * y_pred_c, axis=-1)
        return -tf_module.gather(alpha, y_true_i) * tf_module.pow(1.0 - p_t, gamma) * tf_module.math.log(p_t)
    return loss


def compile_model(model, args: argparse.Namespace, alpha_values, tf_module, learning_rate: float) -> None:
    losses, metrics, weights = {}, {}, {}
    if args.task in ("classification", "multitask"):
        losses["class_output"] = sparse_focal_loss(alpha_values, gamma=1.5, tf_module=tf_module)
        metrics["class_output"] = ["accuracy"]
        weights["class_output"] = 1.0
    if args.task in ("regression", "multitask"):
        losses["bilirubin_output"] = tf_module.keras.losses.Huber(delta=1.0)
        metrics["bilirubin_output"] = ["mae"]
        weights["bilirubin_output"] = 0.35 if args.task == "multitask" else 1.0
    model.compile(optimizer=tf_module.keras.optimizers.Adam(learning_rate), loss=losses, loss_weights=weights, metrics=metrics)


def augmentation_layers(tf_module, name: str):
    L = tf_module.keras.layers
    return tf_module.keras.Sequential([
        L.RandomFlip("horizontal", name=f"{name}_flip"),
        L.RandomRotation(0.025, fill_mode="nearest", name=f"{name}_rotate"),
        L.RandomZoom(0.05, fill_mode="nearest", name=f"{name}_zoom"),
        L.RandomTranslation(0.03, 0.03, fill_mode="nearest", name=f"{name}_translate"),
    ], name=name)


def compact_2d_features(image_input, args: argparse.Namespace, tf_module):
    L = tf_module.keras.layers
    x = L.Rescaling(1.0 / 255.0)(image_input)
    x = augmentation_layers(tf_module, "aug")(x)
    for filters, dropout in ((32, 0.05), (64, 0.10), (128, 0.15), (192, 0.20)):
        x = L.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("swish")(x)
        x = L.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("swish")(x)
        x = L.MaxPooling2D(2)(x)
        x = L.SpatialDropout2D(dropout)(x)
    x = L.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)
    x = L.GlobalAveragePooling2D()(x)
    return L.Dense(256, activation="swish")(x)


def mobilenetv2_features(image_input, args: argparse.Namespace, tf_module):
    L, apps = tf_module.keras.layers, tf_module.keras.applications
    x = augmentation_layers(tf_module, "aug")(image_input)
    rgb = L.Lambda(lambda z: z[..., :3])(x)
    hsv = L.Lambda(lambda z: z[..., 3:])(x)
    weights = "imagenet" if args.weights == "imagenet" else None
    try:
        backbone = apps.MobileNetV2(include_top=False, weights=weights, input_shape=(args.image_size, args.image_size, 3), pooling="avg")
    except Exception:
        backbone = apps.MobileNetV2(include_top=False, weights=None, input_shape=(args.image_size, args.image_size, 3), pooling="avg")
    backbone.trainable = False
    rgb_features = backbone(L.Lambda(apps.mobilenet_v2.preprocess_input)(rgb))
    h = L.Rescaling(1.0/255.0)(hsv)
    for f in (32, 64, 128):
        h = L.Conv2D(f, 3, padding="same", activation="swish")(h)
        h = L.MaxPooling2D(2)(h)
    return L.Concatenate()([rgb_features, L.GlobalAveragePooling2D()(h)])


def build_model(args: argparse.Namespace, alpha_values, use_metadata: bool, tf_module):
    L = tf_module.keras.layers
    img_in = tf_module.keras.Input(shape=(args.image_size, args.image_size, 6), name="image_input")
    x = compact_2d_features(img_in, args, tf_module) if args.model == "compact_2d" else mobilenetv2_features(img_in, args, tf_module)
    inputs = [img_in]
    if use_metadata:
        meta_in = tf_module.keras.Input(shape=(len(META_COLUMNS),), name="metadata_input")
        m = L.Dense(32, activation="swish")(meta_in)
        m = L.BatchNormalization()(m)
        m = L.Dropout(0.15)(m)
        m = L.Dense(16, activation="swish")(m)
        x = L.Concatenate()([x, m])
        inputs.append(meta_in)
    x = L.Dense(256, activation="swish")(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.35)(x)
    x = L.Dense(128, activation="swish")(x)
    x = L.Dropout(0.25)(x)
    outputs = {}
    if args.task in ("classification", "multitask"):
        outputs["class_output"] = L.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)
    if args.task in ("regression", "multitask"):
        outputs["bilirubin_output"] = L.Dense(1, activation="linear", name="bilirubin_output")(x)
    model = tf_module.keras.Model(inputs=inputs, outputs=outputs)
    compile_model(model, args, alpha_values, tf_module, args.learning_rate)
    return model


def make_inputs(X, meta_scaled, indices, use_meta: bool):
    return {"image_input": X[indices], "metadata_input": meta_scaled[indices]} if use_meta else {"image_input": X[indices]}


def make_targets(y_cls, y_reg_norm, indices, task: str):
    t = {}
    if task in ("classification", "multitask"): t["class_output"] = y_cls[indices]
    if task in ("regression", "multitask"): t["bilirubin_output"] = y_reg_norm[indices]
    return t


def callbacks_for_fold(args: argparse.Namespace, fold_dir: Path, tf_module):
    fold_dir.mkdir(parents=True, exist_ok=True)
    p = 3 if args.smoke else args.patience
    return [
        tf_module.keras.callbacks.EarlyStopping(monitor="val_loss", patience=p, mode="min", restore_best_weights=True, verbose=1),
        tf_module.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, p//2), min_lr=1e-7, mode="min", verbose=1),
        tf_module.keras.callbacks.CSVLogger(str(fold_dir / "history.csv")),
        tf_module.keras.callbacks.ModelCheckpoint(str(fold_dir / "best_model.keras"), monitor="val_loss", mode="min", save_best_only=True, verbose=0),
    ]


def fine_tune_if_requested(model, args: argparse.Namespace, alpha_values, train_data, val_data, tf_module):
    if args.model != "mobilenetv2" or args.fine_tune_epochs <= 0 or args.smoke: return None
    bb = [l for l in model.layers if "mobilenetv2" in l.name.lower()][0]
    bb.trainable = True
    if args.fine_tune_layers > 0:
        for layer in bb.layers[:-args.fine_tune_layers]: layer.trainable = False
    compile_model(model, args, alpha_values, tf_module, args.fine_tune_lr)
    print(f"Fine-tuning {args.fine_tune_layers} layers for {args.fine_tune_epochs} epochs.")
    return model.fit(train_data["x"], train_data["y"], epochs=args.fine_tune_epochs, batch_size=args.batch_size, validation_data=(val_data["x"], val_data["y"]), verbose=2)


def run_fold(fold_num, train_idx, test_idx, X, y_cls, y_reg, groups, metadata, samples, args, deps):
    np_m, tf_m = deps["np"], deps["tf"]
    use_meta = not args.no_metadata
    train_idx, val_idx = validation_split(train_idx, y_cls, groups, args, deps)
    meta_scaled = metadata_transform(metadata, train_idx, np_m)
    y_reg_n, r_mean, r_std = regression_transform(y_reg, train_idx, np_m)
    alpha = class_alpha(y_cls[train_idx], np_m)

    fold_dir = args.output_dir / "folds" / f"fold_{fold_num:02d}"
    model = build_model(args, alpha, use_meta, tf_m)
    if fold_num == 1: model.summary()

    tr_x, val_x, ts_x = make_inputs(X, meta_scaled, train_idx, use_meta), make_inputs(X, meta_scaled, val_idx, use_meta), make_inputs(X, meta_scaled, test_idx, use_meta)
    tr_y, val_y = make_targets(y_cls, y_reg_n, train_idx, args.task), make_targets(y_cls, y_reg_n, val_idx, args.task)

    print(f"\nFold {fold_num}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    hist = model.fit(tr_x, tr_y, validation_data=(val_x, val_y), epochs=2 if args.smoke else args.epochs, batch_size=args.batch_size, callbacks=callbacks_for_fold(args, fold_dir, tf_m), verbose=2)
    fine_tune_if_requested(model, args, alpha, {"x": tr_x, "y": tr_y}, {"x": val_x, "y": val_y}, tf_m)

    preds = model.predict(ts_x, batch_size=args.batch_size, verbose=0)
    pred_dict = preds if isinstance(preds, dict) else ({k: v for k, v in zip(model.output_names, preds)} if isinstance(preds, list) else {model.output_names[0]: preds})

    res = {"fold": fold_num, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))}
    if args.task in ("classification", "multitask"):
        probs = pred_dict["class_output"]
        p_cls = np_m.argmax(probs, axis=1)
        res.update({
            "accuracy": float(deps["accuracy_score"](y_cls[test_idx], p_cls) * 100.0),
            "f1_weighted": float(deps["f1_score"](y_cls[test_idx], p_cls, average="weighted", zero_division=0) * 100.0)
        })
    if args.task in ("regression", "multitask"):
        p_reg = pred_dict["bilirubin_output"].reshape(-1) * r_std + r_mean
        res["regression_mae"] = float(deps["mean_absolute_error"](y_reg[test_idx], p_reg))

    model.save(fold_dir / "final_model.keras")
    with (fold_dir / "metrics.json").open("w") as h: json.dump(res, h, indent=2)
    return res


def main() -> None:
    args = parse_args()
    validate_paths(args)
    deps = import_dependencies()
    np_m, tf_m = deps["np"], deps["tf"]

    set_reproducibility(args.seed, tf_m)
    configure_accelerator(tf_m)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args, np_m)
    X = load_or_build_cache(samples, args, deps)
    y_cls, y_reg, groups, meta = raw_arrays(samples, np_m)
    folds = make_folds(y_cls, groups, args, deps)

    results = [run_fold(i, np_m.asarray(tr), np_m.asarray(ts), X, y_cls, y_reg, groups, meta, samples, args, deps) for i, (tr, ts) in enumerate(folds, start=1)]
    
    agg = {}
    for k in ["accuracy", "f1_weighted", "regression_mae"]:
        vals = [r[k] for r in results if k in r]
        if vals: agg[f"mean_{k}"] = sum(vals)/len(vals)
    
    with (args.output_dir / "summary.json").open("w") as h: json.dump({"folds": results, "aggregate": agg}, h, indent=2)
    print(f"\nTraining Complete. Summary: {args.output_dir / 'summary.json'}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
