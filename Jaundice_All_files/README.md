# Neonatal Jaundice Screening — Multi-Modal Deep Learning

A multi-modal deep learning pipeline for non-invasive neonatal jaundice screening using skin images and clinical parameters.

## Overview

This project implements a **multi-modal fusion model** that combines:
- **Visual Branch**: 1D-CNN processing skin pixel color sequences (RGB + HSV) from neonatal images
- **Clinical Branch**: MLP processing clinical features (weight, gestational age, postnatal age, gender, treatment)
- **Fusion**: Concatenated embeddings → fully connected layers → dual output heads

### Dual Output Heads
1. **Regression**: Predicts TSB (Total Serum Bilirubin) in mg/dL
2. **Classification**: Predicts risk category (Low / Medium / High)

## Project Structure

```
├── development/
│   ├── multimodal_fusion_v2.ipynb    # Main pipeline (v2 — final)
│   ├── multimodal_fusion.ipynb       # Earlier version (v1)
│   ├── clinical_branch.ipynb         # Clinical-only branch experiments
│   ├── visual_branch.ipynb           # Visual-only branch experiments
│   ├── model_approach.ipynb          # Initial approach exploration
│   ├── evaluation_results_v2.json    # Final evaluation metrics
│   └── *.h5 / *.keras / *.pkl        # Saved models & preprocessing (gitignored)
├── code/
│   ├── model.ipynb                   # Standalone model notebook
│   ├── roi.py                        # ROI extraction utilities
│   └── step_1.py                     # Preprocessing step 1
├── NeoJaundice/
│   └── chd_jaundice_published_2.csv  # Clinical dataset
├── images/                           # Neonatal skin images (gitignored)
└── .gitignore
```

## Dataset

- **NeoJaundice** dataset: 745 patients × 3 images each = 2,235 samples
- Clinical CSV: `NeoJaundice/chd_jaundice_published_2.csv`
- Images: 3 skin photographs per neonate (forehead, chest, limb)

## Key Technical Decisions

| Feature | Implementation |
|---------|---------------|
| Image preprocessing | LAB white balance → HSV skin segmentation → RGB+HSV pixel sequence |
| Data splitting | Patient-level stratified split (prevents leakage) |
| Clinical normalization | MinMaxScaler fit on training data only |
| Class imbalance | Weighted sparse categorical crossentropy loss |
| Architecture | 4-block 1D-CNN + MLP, ~4.4M parameters |
| Training | AdamW (lr=5e-4, wd=1e-4), EarlyStopping, ReduceLR |

## Results (v2)

| Metric | Value |
|--------|-------|
| RMSE | 3.37 mg/dL |
| MAE | 2.77 mg/dL |
| R² | 0.614 |
| Classification Accuracy | 77.3% |
| Weighted F1 | 0.684 |

## Requirements

- Python 3.10+
- TensorFlow 2.16+
- OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn

## Authors

Shubh Bhateja
