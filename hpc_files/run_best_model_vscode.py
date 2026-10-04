#!/usr/bin/env python3
"""
Run the best NeoJaundice model from VSCode.

How to run:
  1. Open this folder in VSCode:
     /Users/deepikasaini/Developer/Shubh_Neonatal_Jaundice/Jaundice_All_files copy
  2. Select the Python interpreter:
     .venv-makhloughi/bin/python
  3. Open this file and press Run.

This uses the best-tested local setup so far:
  - MobileNetV2 ImageNet RGB branch
  - HSV CNN branch
  - metadata fusion
  - original_cropped_skin_patches dataset
  - NeoJaundice/chd_jaundice_published_2.csv labels
  - image-level holdout split

Last local result:
  Accuracy: 75.17%
  Weighted F1: 74.84%
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "NeoJaundice"
CSV_PATH = DATA_DIR / "chd_jaundice_published_2.csv"
IMAGE_DIR = PROJECT_ROOT / "original_cropped_skin_patches"
OUTPUT_DIR = PROJECT_ROOT / "vscode_best_model_outputs"


def validate_paths() -> None:
    missing = []
    if not CSV_PATH.exists():
        missing.append(str(CSV_PATH))
    if not IMAGE_DIR.exists():
        missing.append(str(IMAGE_DIR))

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Required dataset files/folders are missing:\n"
            f"  {joined}\n\n"
            "Make sure you are running this file from the project folder that contains "
            "NeoJaundice/ and original_cropped_skin_patches/."
        )


def main() -> None:
    validate_paths()
    os.chdir(PROJECT_ROOT)

    from sota_2d_neojaundice import main as train_main

    sys.argv = [
        sys.argv[0],
        "--model",
        "mobilenetv2",
        "--weights",
        "imagenet",
        "--task",
        "classification",
        "--split",
        "image-holdout",
        "--data-dir",
        str(DATA_DIR),
        "--csv-path",
        str(CSV_PATH),
        "--image-dir",
        str(IMAGE_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
        "--epochs",
        "25",
        "--patience",
        "6",
        "--fine-tune-epochs",
        "0",
        "--batch-size",
        "32",
        "--image-size",
        "128",
    ]
    train_main()


if __name__ == "__main__":
    main()
