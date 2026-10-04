#!/usr/bin/env python3
"""
SOTA-oriented 2D NeoJaundice experiments.

This script intentionally does not replace makhloughi_exact_replication.py.
It is for performance-seeking experiments beyond the disclosed 1D-CNN paper
pipeline:

  - Uses real 2D skin patches instead of flattened 1D color series.
  - Builds RGB+HSV six-channel image tensors.
  - Can fuse safe clinical metadata: gender, gestational age, age, weight.
  - Supports compact 2D-CNN and MobileNetV2 transfer-learning backbones.
  - Supports image-level and patient-level folds. Patient-level is the honest
    generalization protocol because each patient has three images.
  - Saves predictions, histories, confusion matrices, trained models, and JSON
    metrics for every fold.

The goal is to make stronger models easy to compare without accidentally
mixing paper replication, leakage-prone splits, and SOTA experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "NeoJaundice"
DEFAULT_CROPPED_DIR = PROJECT_ROOT / "original_cropped_skin_patches"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sota_2d_outputs"

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
    default_image_dir = DEFAULT_CROPPED_DIR if DEFAULT_CROPPED_DIR.exists() else DEFAULT_DATA_DIR / "images"
    parser = argparse.ArgumentParser(
        description="Train stronger 2D CNN / transfer-learning models on NeoJaundice."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("compact_2d", "mobilenetv2"),
        default="compact_2d",
        help="compact_2d is CPU-friendly; mobilenetv2 is the stronger transfer-learning option.",
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "multitask"),
        default="multitask",
        help="multitask trains bilirubin regression and 3-class classification together.",
    )
    parser.add_argument(
        "--split",
        choices=("patient-kfold", "image-kfold", "patient-holdout", "image-holdout"),
        default="patient-kfold",
        help="patient-* avoids leakage across the three images per patient.",
    )
    parser.add_argument(
        "--calibration",
        choices=("auto", "none", "gray-world", "yellow-card"),
        default="auto",
        help="auto uses no calibration for cropped patches and yellow-card calibration for full images.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small balanced subset and short training for component verification.",
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
    if value < 5.0:
        return 0
    if value < 10.0:
        return 1
    return 2


def encode_gender(value: str) -> float:
    text = str(value).strip().upper()
    if text == "M":
        return 1.0
    if text == "F":
        return 0.0
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
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

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
        raise ValueError("No image rows could be loaded.")

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
    contours, _ = cv2_module.findContours(
        mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return bgr

    largest = max(contours, key=cv2_module.contourArea)
    if cv2_module.contourArea(largest) < 200:
        return bgr

    card_mask = np_module.zeros(mask.shape, dtype=np_module.uint8)
    cv2_module.drawContours(card_mask, [largest], contourIdx=-1, color=255, thickness=-1)

    lab = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2LAB).astype(np_module.float32)
    mean_lab = cv2_module.mean(lab, mask=card_mask)[:3]
    correction = np_module.array(YELLOW_CARD_REFERENCE_OPENCV, dtype=np_module.float32) - np_module.array(
        mean_lab, dtype=np_module.float32
    )
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
    # NeoJaundice full images include the color card near the lower area.
    mask[int(height * 0.82) :, :] = 0
    kernel = cv2_module.getStructuringElement(cv2_module.MORPH_ELLIPSE, (7, 7))
    mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_OPEN, kernel, iterations=2)
    mask = cv2_module.morphologyEx(mask, cv2_module.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2_module.findContours(mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    if not contours:
        size = min(height, width)
        y1 = (height - size) // 2
        x1 = (width - size) // 2
        return bgr[y1 : y1 + size, x1 : x1 + size]

    contour = max(contours, key=cv2_module.contourArea)
    x, y, w, h = cv2_module.boundingRect(contour)
    inset_x = int(w * 0.05)
    inset_y = int(h * 0.05)
    x1 = max(0, x + inset_x)
    y1 = max(0, y + inset_y)
    x2 = min(width, x + w - inset_x)
    y2 = min(height, y + h - inset_y)
    if x2 <= x1 or y2 <= y1:
        return bgr
    return bgr[y1:y2, x1:x2]


def choose_calibration(mode: str, image_dir: Path, bgr_shape: tuple[int, int, int]) -> str:
    if mode != "auto":
        return mode
    if image_dir.name == "original_cropped_skin_patches":
        return "none"
    if bgr_shape[0] >= 500 and bgr_shape[1] >= 500:
        return "yellow-card"
    return "none"


def preprocess_image(sample: Sample, args: argparse.Namespace, deps):
    cv2_module = deps["cv2"]
    np_module = deps["np"]

    bgr = cv2_module.imread(str(sample.image_path), cv2_module.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {sample.image_path}")

    calibration = choose_calibration(args.calibration, args.image_dir, bgr.shape)
    if calibration == "yellow-card":
        bgr = yellow_card_white_balance(bgr, cv2_module, np_module)
    elif calibration == "gray-world":
        bgr = gray_world_lab_balance(bgr, cv2_module, np_module)

    if bgr.shape[0] >= 500 and bgr.shape[1] >= 500 and args.image_dir.name != "original_cropped_skin_patches":
        bgr = adaptive_skin_crop_from_full_image(bgr, cv2_module, np_module)

    bgr = cv2_module.resize(bgr, (args.image_size, args.image_size), interpolation=cv2_module.INTER_AREA)
    rgb = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    hsv = cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2HSV)
    hsv = hsv.copy()
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(parents=True, exist_ok=True)
    x_path = cache_file(args)
    labels_path = labels_cache_file(args)

    image_names = [sample.image_idx for sample in samples]
    if not args.rebuild_cache and x_path.exists() and labels_path.exists():
        with labels_path.open() as handle:
            payload = json.load(handle)
        if payload.get("image_idx") == image_names and payload.get("image_size") == args.image_size:
            print(f"Loading image cache: {x_path}")
            X = np_module.load(x_path, mmap_mode=None)
            return X

    print("Building 2D RGB+HSV image cache.")
    start = time.time()
    X = np_module.zeros(
        (len(samples), args.image_size, args.image_size, 6),
        dtype=np_module.uint8,
    )
    for i, sample in enumerate(samples):
        X[i] = preprocess_image(sample, args, deps)
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(samples):
            print(f"  processed {i + 1:4d}/{len(samples)} images ({time.time() - start:.1f}s)")

    np_module.save(x_path, X)
    with labels_path.open("w") as handle:
        json.dump(
            {
                "image_idx": image_names,
                "patient_id": [sample.patient_id for sample in samples],
                "bilirubin": [sample.bilirubin for sample in samples],
                "class_id": [sample.class_id for sample in samples],
                "image_size": args.image_size,
            },
            handle,
            indent=2,
        )
    return X


def raw_arrays(samples: list[Sample], np_module):
    y_cls = np_module.array([sample.class_id for sample in samples], dtype=np_module.int64)
    y_reg = np_module.array([sample.bilirubin for sample in samples], dtype=np_module.float32)
    groups = np_module.array([sample.patient_id for sample in samples])
    metadata = np_module.array([sample.metadata for sample in samples], dtype=np_module.float32)
    return y_cls, y_reg, groups, metadata


def make_folds(y_cls, groups, args: argparse.Namespace, deps):
    np_module = deps["np"]
    StratifiedKFold = deps["StratifiedKFold"]
    StratifiedGroupKFold = deps["StratifiedGroupKFold"]
    StratifiedShuffleSplit = deps["StratifiedShuffleSplit"]

    n_splits = 2 if args.smoke else args.folds
    if args.split == "image-kfold":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        return list(splitter.split(np_module.zeros(len(y_cls)), y_cls))

    if args.split == "patient-kfold":
        if StratifiedGroupKFold is None:
            raise RuntimeError("Installed scikit-learn does not provide StratifiedGroupKFold.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        return list(splitter.split(np_module.zeros(len(y_cls)), y_cls, groups))

    if args.split == "image-holdout":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        return list(splitter.split(np_module.zeros(len(y_cls)), y_cls))

    if args.split == "patient-holdout":
        if StratifiedGroupKFold is None:
            raise RuntimeError("Installed scikit-learn does not provide StratifiedGroupKFold.")
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(np_module.zeros(len(y_cls)), y_cls, groups))
        return [(train_idx, test_idx)]

    raise ValueError(f"Unknown split: {args.split}")


def validation_split(train_idx, y_cls, groups, args: argparse.Namespace, deps):
    np_module = deps["np"]
    StratifiedGroupKFold = deps["StratifiedGroupKFold"]
    StratifiedShuffleSplit = deps["StratifiedShuffleSplit"]

    if args.split.startswith("patient"):
        if StratifiedGroupKFold is None:
            raise RuntimeError("Installed scikit-learn does not provide StratifiedGroupKFold.")
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 101)
        local = next(
            splitter.split(
                np_module.zeros(len(train_idx)),
                y_cls[train_idx],
                groups[train_idx],
            )
        )
        tr_local, val_local = local
        return train_idx[tr_local], train_idx[val_local]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=args.seed + 101)
    tr_local, val_local = next(splitter.split(np_module.zeros(len(train_idx)), y_cls[train_idx]))
    return train_idx[tr_local], train_idx[val_local]


def metadata_transform(metadata, train_idx, np_module):
    train_meta = metadata[train_idx].astype(np_module.float32)
    medians = np_module.nanmedian(train_meta, axis=0)
    medians = np_module.where(np_module.isfinite(medians), medians, 0.0)
    filled = np_module.where(np_module.isnan(metadata), medians.reshape(1, -1), metadata)
    train_filled = filled[train_idx]
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0)
    std = np_module.where(std < 1e-6, 1.0, std)
    return ((filled - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np_module.float32)


def regression_transform(y_reg, train_idx, np_module):
    mean = float(y_reg[train_idx].mean())
    std = float(y_reg[train_idx].std())
    if std < 1e-6:
        std = 1.0
    y_norm = ((y_reg - mean) / std).astype(np_module.float32)
    return y_norm, mean, std


def class_alpha(y_train, np_module):
    counts = np_module.bincount(y_train, minlength=NUM_CLASSES).astype(np_module.float32)
    weights = counts.sum() / (NUM_CLASSES * np_module.maximum(counts, 1.0))
    return (weights / weights.mean()).astype(np_module.float32)


def sparse_focal_loss(alpha_values, gamma: float, tf_module):
    alpha = tf_module.constant(alpha_values, dtype=tf_module.float32)

    def loss(y_true, y_pred):
        y_true_i = tf_module.cast(tf_module.reshape(y_true, [-1]), tf_module.int32)
        y_pred_c = tf_module.clip_by_value(
            y_pred,
            tf_module.keras.backend.epsilon(),
            1.0 - tf_module.keras.backend.epsilon(),
        )
        one_hot = tf_module.one_hot(y_true_i, depth=NUM_CLASSES)
        p_t = tf_module.reduce_sum(one_hot * y_pred_c, axis=-1)
        alpha_t = tf_module.gather(alpha, y_true_i)
        return -alpha_t * tf_module.pow(1.0 - p_t, gamma) * tf_module.math.log(p_t)

    return loss


def compile_model(model, args: argparse.Namespace, alpha_values, tf_module, learning_rate: float) -> None:
    losses = {}
    metrics = {}
    loss_weights = {}

    if args.task in ("classification", "multitask"):
        losses["class_output"] = sparse_focal_loss(alpha_values, gamma=1.5, tf_module=tf_module)
        metrics["class_output"] = ["accuracy"]
        loss_weights["class_output"] = 1.0

    if args.task in ("regression", "multitask"):
        losses["bilirubin_output"] = tf_module.keras.losses.Huber(delta=1.0)
        metrics["bilirubin_output"] = ["mae"]
        loss_weights["bilirubin_output"] = 0.35 if args.task == "multitask" else 1.0

    model.compile(
        optimizer=tf_module.keras.optimizers.Adam(learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )


def augmentation_layers(tf_module, name: str):
    layers = tf_module.keras.layers
    return tf_module.keras.Sequential(
        [
            layers.RandomFlip("horizontal", name=f"{name}_flip"),
            layers.RandomRotation(0.025, fill_mode="nearest", name=f"{name}_rotate"),
            layers.RandomZoom(0.05, fill_mode="nearest", name=f"{name}_zoom"),
            layers.RandomTranslation(0.03, 0.03, fill_mode="nearest", name=f"{name}_translate"),
        ],
        name=name,
    )


def compact_2d_features(image_input, args: argparse.Namespace, tf_module):
    layers = tf_module.keras.layers
    x = layers.Rescaling(1.0 / 255.0, name="image_rescale")(image_input)
    x = augmentation_layers(tf_module, "safe_geometric_augmentation")(x)

    for filters, dropout in ((32, 0.05), (64, 0.10), (128, 0.15), (192, 0.20)):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(dropout)(x)

    x = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dense(256, activation="swish", name="compact_image_embedding")(x)


def mobilenetv2_features(image_input, args: argparse.Namespace, tf_module):
    layers = tf_module.keras.layers
    applications = tf_module.keras.applications

    x = augmentation_layers(tf_module, "safe_geometric_augmentation")(image_input)
    rgb = layers.Lambda(lambda z: z[..., :3], name="rgb_slice")(x)
    hsv = layers.Lambda(lambda z: z[..., 3:], name="hsv_slice")(x)

    weights = None if args.weights == "none" else "imagenet"
    try:
        backbone = applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=(args.image_size, args.image_size, 3),
            pooling="avg",
            name="mobilenetv2_rgb_backbone",
        )
    except Exception as exc:
        if weights != "imagenet":
            raise
        print(f"Could not load ImageNet weights ({exc}); falling back to random MobileNetV2 weights.")
        backbone = applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(args.image_size, args.image_size, 3),
            pooling="avg",
            name="mobilenetv2_rgb_backbone",
        )
    backbone.trainable = False

    rgb_preprocessed = layers.Lambda(
        applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(rgb)
    rgb_features = backbone(rgb_preprocessed)

    hsv_norm = layers.Rescaling(1.0 / 255.0, name="hsv_rescale")(hsv)
    h = layers.Conv2D(32, 3, padding="same", activation="swish")(hsv_norm)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Conv2D(64, 3, padding="same", activation="swish")(h)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Conv2D(128, 3, padding="same", activation="swish")(h)
    h = layers.GlobalAveragePooling2D()(h)

    return layers.Concatenate(name="rgb_hsv_transfer_embedding")([rgb_features, h])


def build_model(args: argparse.Namespace, alpha_values, use_metadata: bool, tf_module):
    layers = tf_module.keras.layers
    image_input = tf_module.keras.Input(
        shape=(args.image_size, args.image_size, 6),
        name="image_input",
        dtype=tf_module.float32,
    )

    if args.model == "compact_2d":
        features = compact_2d_features(image_input, args, tf_module)
    elif args.model == "mobilenetv2":
        features = mobilenetv2_features(image_input, args, tf_module)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    inputs = [image_input]
    x = features
    if use_metadata:
        meta_input = tf_module.keras.Input(shape=(len(META_COLUMNS),), name="metadata_input")
        m = layers.Dense(32, activation="swish")(meta_input)
        m = layers.BatchNormalization()(m)
        m = layers.Dropout(0.15)(m)
        m = layers.Dense(16, activation="swish")(m)
        x = layers.Concatenate(name="image_metadata_fusion")([x, m])
        inputs.append(meta_input)

    x = layers.Dense(256, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.25)(x)

    outputs = {}

    if args.task in ("classification", "multitask"):
        outputs["class_output"] = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

    if args.task in ("regression", "multitask"):
        outputs["bilirubin_output"] = layers.Dense(1, activation="linear", name="bilirubin_output")(x)

    model = tf_module.keras.Model(inputs=inputs, outputs=outputs, name=f"neojaundice_{args.model}_{args.task}")
    compile_model(model, args, alpha_values, tf_module, args.learning_rate)
    return model


def make_inputs(X, metadata_scaled, indices, use_metadata: bool):
    if use_metadata:
        return {
            "image_input": X[indices],
            "metadata_input": metadata_scaled[indices],
        }
    return {"image_input": X[indices]}


def make_targets(y_cls, y_reg_norm, indices, task: str):
    targets = {}
    if task in ("classification", "multitask"):
        targets["class_output"] = y_cls[indices]
    if task in ("regression", "multitask"):
        targets["bilirubin_output"] = y_reg_norm[indices]
    return targets


def make_sample_weights(y_cls, indices, alpha_values, task: str, np_module):
    # Class imbalance is handled inside sparse_focal_loss via alpha_values.
    # Avoid applying the same class weights twice.
    return None


def callbacks_for_fold(args: argparse.Namespace, fold_dir: Path, tf_module):
    fold_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val_loss"
    return [
        tf_module.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=3 if args.smoke else args.patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf_module.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=2 if args.smoke else max(3, args.patience // 2),
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
        tf_module.keras.callbacks.CSVLogger(str(fold_dir / "history.csv")),
        tf_module.keras.callbacks.ModelCheckpoint(
            str(fold_dir / "best_model.keras"),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
    ]


def fine_tune_if_requested(model, args: argparse.Namespace, alpha_values, train_data, val_data, tf_module):
    if args.model != "mobilenetv2" or args.fine_tune_epochs <= 0 or args.smoke:
        return None

    backbone = model.get_layer("mobilenetv2_rgb_backbone")
    backbone.trainable = True
    if args.fine_tune_layers > 0:
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False

    compile_model(model, args, alpha_values, tf_module, args.fine_tune_lr)
    print(f"Fine-tuning last {args.fine_tune_layers} MobileNetV2 layers for {args.fine_tune_epochs} epochs.")
    return model.fit(
        train_data["x"],
        train_data["y"],
        sample_weight=train_data["sample_weight"],
        validation_data=(val_data["x"], val_data["y"]),
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        verbose=2,
    )


def classification_metrics(y_true, y_pred, deps):
    return {
        "accuracy": float(deps["accuracy_score"](y_true, y_pred) * 100.0),
        "balanced_accuracy": float(deps["balanced_accuracy_score"](y_true, y_pred) * 100.0),
        "precision_weighted": float(
            deps["precision_score"](y_true, y_pred, average="weighted", zero_division=0) * 100.0
        ),
        "recall_weighted": float(
            deps["recall_score"](y_true, y_pred, average="weighted", zero_division=0) * 100.0
        ),
        "f1_weighted": float(deps["f1_score"](y_true, y_pred, average="weighted", zero_division=0) * 100.0),
        "f1_macro": float(deps["f1_score"](y_true, y_pred, average="macro", zero_division=0) * 100.0),
    }


def regression_metrics(y_true, y_pred, deps):
    mse = deps["mean_squared_error"](y_true, y_pred)
    return {
        "mae": float(deps["mean_absolute_error"](y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(deps["r2_score"](y_true, y_pred)),
    }


def write_predictions(
    path: Path,
    samples: list[Sample],
    indices,
    y_cls,
    pred_cls,
    pred_probs,
    y_reg,
    pred_reg,
    np_module,
) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = [
            "patient_id",
            "image_idx",
            "true_bilirubin",
            "true_class",
            "pred_class",
            "prob_0_5",
            "prob_5_10",
            "prob_gt_10",
            "pred_bilirubin",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_i, sample_i in enumerate(indices):
            sample = samples[int(sample_i)]
            probs = pred_probs[row_i] if pred_probs is not None else np_module.full(NUM_CLASSES, np_module.nan)
            writer.writerow(
                {
                    "patient_id": sample.patient_id,
                    "image_idx": sample.image_idx,
                    "true_bilirubin": float(y_reg[sample_i]),
                    "true_class": CLASS_NAMES[int(y_cls[sample_i])],
                    "pred_class": CLASS_NAMES[int(pred_cls[row_i])] if pred_cls is not None else "",
                    "prob_0_5": float(probs[0]),
                    "prob_5_10": float(probs[1]),
                    "prob_gt_10": float(probs[2]),
                    "pred_bilirubin": float(pred_reg[row_i]) if pred_reg is not None else "",
                }
            )


def run_fold(
    fold_number: int,
    train_idx,
    test_idx,
    X,
    y_cls,
    y_reg,
    groups,
    metadata,
    samples: list[Sample],
    args: argparse.Namespace,
    deps,
):
    np_module = deps["np"]
    tf_module = deps["tf"]
    use_metadata = not args.no_metadata

    train_idx, val_idx = validation_split(train_idx, y_cls, groups, args, deps)
    metadata_scaled = metadata_transform(metadata, train_idx, np_module)
    y_reg_norm, reg_mean, reg_std = regression_transform(y_reg, train_idx, np_module)
    alpha_values = class_alpha(y_cls[train_idx], np_module)

    fold_dir = args.output_dir / "folds" / f"fold_{fold_number:02d}"
    model = build_model(args, alpha_values, use_metadata, tf_module)
    if fold_number == 1:
        model.summary()

    train_x = make_inputs(X, metadata_scaled, train_idx, use_metadata)
    val_x = make_inputs(X, metadata_scaled, val_idx, use_metadata)
    test_x = make_inputs(X, metadata_scaled, test_idx, use_metadata)
    train_y = make_targets(y_cls, y_reg_norm, train_idx, args.task)
    val_y = make_targets(y_cls, y_reg_norm, val_idx, args.task)
    sample_weight = make_sample_weights(y_cls, train_idx, alpha_values, args.task, np_module)

    epochs = 2 if args.smoke else args.epochs
    print(
        f"\nFold {fold_number}: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, alpha={alpha_values.tolist()}"
    )
    history = model.fit(
        train_x,
        train_y,
        sample_weight=sample_weight,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=args.batch_size,
        callbacks=callbacks_for_fold(args, fold_dir, tf_module),
        verbose=2,
    )

    fine_tune_if_requested(
        model,
        args,
        alpha_values,
        {"x": train_x, "y": train_y, "sample_weight": sample_weight},
        {"x": val_x, "y": val_y},
        tf_module,
    )

    predictions = model.predict(test_x, batch_size=args.batch_size, verbose=0)
    if isinstance(predictions, dict):
        pred_dict = predictions
    elif isinstance(predictions, list):
        keys = list(model.output_names)
        pred_dict = {key: value for key, value in zip(keys, predictions)}
    else:
        pred_dict = {model.output_names[0]: predictions}

    result = {
        "fold": fold_number,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs_completed": int(len(history.history.get("loss", []))),
        "regression_mean": reg_mean,
        "regression_std": reg_std,
    }

    pred_cls = None
    pred_probs = None
    if args.task in ("classification", "multitask"):
        pred_probs = pred_dict["class_output"]
        pred_cls = np_module.argmax(pred_probs, axis=1)
        result.update(classification_metrics(y_cls[test_idx], pred_cls, deps))
        cm = deps["confusion_matrix"](y_cls[test_idx], pred_cls, labels=[0, 1, 2])
        np_module.savetxt(fold_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    pred_reg = None
    if args.task in ("regression", "multitask"):
        pred_reg_norm = pred_dict["bilirubin_output"].reshape(-1)
        pred_reg = pred_reg_norm * reg_std + reg_mean
        result.update({f"regression_{k}": v for k, v in regression_metrics(y_reg[test_idx], pred_reg, deps).items()})

    write_predictions(
        fold_dir / "predictions.csv",
        samples,
        test_idx,
        y_cls,
        pred_cls,
        pred_probs,
        y_reg,
        pred_reg,
        np_module,
    )
    model.save(fold_dir / "final_model.keras")

    with (fold_dir / "metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Fold {fold_number} metrics: {json.dumps(result, indent=2)}")
    return result


def aggregate_results(fold_results: list[dict]) -> dict:
    aggregate: dict[str, float] = {}
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "f1_macro",
        "regression_mae",
        "regression_rmse",
        "regression_r2",
    ]
    for key in metric_keys:
        values = [result[key] for result in fold_results if key in result]
        if values:
            aggregate[f"mean_{key}"] = float(sum(values) / len(values))
            if len(values) > 1:
                mean = aggregate[f"mean_{key}"]
                aggregate[f"std_{key}"] = float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))
    return aggregate


def write_run_config(args: argparse.Namespace, samples: list[Sample]) -> None:
    payload = {
        "data_dir": str(args.data_dir),
        "csv_path": str(resolve_csv_path(args)),
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "calibration": args.calibration,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fine_tune_epochs": args.fine_tune_epochs,
        "weights": args.weights,
        "folds": args.folds,
        "patience": args.patience,
        "seed": args.seed,
        "use_metadata": not args.no_metadata,
        "smoke": args.smoke,
        "sample_count": len(samples),
        "patient_count": len(set(sample.patient_id for sample in samples)),
    }
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    deps = import_dependencies()
    np_module = deps["np"]
    tf_module = deps["tf"]

    set_reproducibility(args.seed, tf_module)
    configure_accelerator(tf_module)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args, np_module)
    write_run_config(args, samples)
    X = load_or_build_cache(samples, args, deps)
    y_cls, y_reg, groups, metadata = raw_arrays(samples, np_module)
    folds = make_folds(y_cls, groups, args, deps)

    fold_results = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_results.append(
            run_fold(
                fold_number,
                np_module.asarray(train_idx),
                np_module.asarray(test_idx),
                X,
                y_cls,
                y_reg,
                groups,
                metadata,
                samples,
                args,
                deps,
            )
        )

    summary = {
        "folds": fold_results,
        "aggregate": aggregate_results(fold_results),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nWrote summary: {args.output_dir / 'summary.json'}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
