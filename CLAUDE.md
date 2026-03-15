# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Skin cancer classification thesis project focused on lesion segmentation using Mask R-CNN, radiomic feature extraction, and ML techniques. Datasets used: HAM10000, ISIC Archive, and a custom UDEM dataset.

## Environment Setup

There are multiple virtual environments for different components (all gitignored):
- `.venv/` — general environment
- `.venv-mask/` — Mask R-CNN training/inference
- `.venv-features/` — feature extraction

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch and torchvision are used in `src/mask/` but are not listed in `requirements.txt` — install them separately as needed.

## Running Notebooks

Notebooks are numbered and meant to be run in order:
- `notebooks/00_eda_isci_archive.ipynb` — EDA on ISIC archive metadata
- `notebooks/01_train_mask_rcnn.ipynb` — Train Mask R-CNN for lesion segmentation
- `notebooks/02_apply_segmentation.ipynb` — Apply trained model to segment lesions

Launch with:
```bash
jupyter notebook notebooks/
```

## Architecture

### Pipeline

1. **EDA** (`notebooks/00_*`) — explores `data/metadata.csv` (ISIC archive) and `data/udem_skin_cancer.csv`
2. **Annotation prep** (`src/mask/coco_annotations.py`) — converts binary masks from HAM10000 to COCO JSON format with RLE encoding
3. **Training** (`notebooks/01_*` + `src/mask/`) — trains Mask R-CNN using torchvision; training loop is in `src/mask/engine.py`
4. **Evaluation** (`src/mask/mask_evaluation.py`) — computes Dice, IoU, Precision, Recall, Specificity, Accuracy; results saved to `results/`
5. **Inference** (`notebooks/02_*`) — loads saved `.pth` model from `models/` and runs segmentation

### `src/mask/` Module

| File | Purpose |
|------|---------|
| `coco_annotations.py` | Converts HAM10000 binary masks → COCO JSON annotations |
| `coco_utils.py` | PyTorch `CocoDetection` dataset class + conversion utilities |
| `coco_eval.py` | `CocoEvaluator` — wraps pycocotools for bbox/segm evaluation |
| `engine.py` | `train_one_epoch()` and `evaluate()` training loops |
| `mask_evaluation.py` | Segmentation metrics: Dice, IoU, Precision, Recall, etc. |
| `transforms.py` | Data augmentation for detection/segmentation (random flip, IoU crop, etc.) |
| `utils.py` | `MetricLogger`, `SmoothedValue`, distributed training helpers |

### Data

- `data/metadata.csv` — ISIC archive metadata (117 MB, tracked in git)
- `data/udem_skin_cancer.csv` — UDEM dataset metadata
- `data/HAM10000/` — raw images and masks (gitignored, must be downloaded separately)
- `models/` — trained `.pth` model files (gitignored)
- `results/` — CSV files with training history and test metrics