# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Skin cancer classification thesis project with a three-stage pipeline:
1. **Lesion segmentation** — Mask R-CNN trained on HAM10000
2. **Radiomic feature extraction** — parallel extraction via pyradiomics
3. **ML/DL classification** — final model trained on extracted features

Datasets: HAM10000 (local), ISIC Archive (GCS), UDEM custom dataset (GCS).

## Python Environments

Three separate environments managed with **pyenv**. All are gitignored.

| Environment | Python | Purpose |
|-------------|--------|---------|
| `.venv-mask/` | 3.12.6 | Mask R-CNN training and inference |
| `.venv-features/` | 3.7 | Radiomic feature extraction (pyradiomics) |
| `.venv/` | latest | Final ML/DL classification model |

Dependencies (fill on the GCP VM after `pip freeze`):
- `requirements/mask.txt` — for `.venv-mask`
- `requirements/features.txt` — for `.venv-features`
- `requirements/ml.txt` — for `.venv`

## EDA Notebooks

EDA notebooks live in `notebooks/eda/`, numbered by dataset in analysis order:

```bash
jupyter notebook notebooks/eda/
```

| Notebook | Dataset | Description |
|----------|---------|-------------|
| `notebooks/eda/01_isic_archive.ipynb` | ISIC Archive | Class distribution, demographics, image dimensions |
| `notebooks/eda/02_ham10000.ipynb` | HAM10000 | Class balance, demographics, mask quality |
| `notebooks/eda/03_udem.ipynb` | UDEM | Custom dataset exploration |

> Training/inference notebooks are archived in `notebooks/archive/` — their logic was extracted to scripts to avoid kernel crashes on GCP remote sessions.

## Scripts (Training & Inference)

Run these in a **tmux session** on GCP to survive SSH disconnections.

| Script | Environment | Description |
|--------|-------------|-------------|
| `scripts/01_train_mask_rcnn.py` | `.venv-mask` | Train Mask R-CNN on HAM10000 |
| `scripts/02_apply_segmentation.py` | `.venv-mask` | Inference on ISIC images → save masks to GCS |
| `scripts/03_extract_features.py` | `.venv-features` | Parallel radiomic feature extraction (resumable) |

## Architecture

### Pipeline

1. **EDA** (`notebooks/eda/`) — explore metadata and class distributions
2. **Annotation prep** (`src/mask/coco_annotations.py`) — convert HAM10000 binary masks → COCO JSON (RLE)
3. **Training** (`scripts/01_train_mask_rcnn.py` + `src/mask/`) — Mask R-CNN via torchvision; loop in `src/mask/engine.py`
4. **Evaluation** (`src/mask/mask_evaluation.py`) — Dice, IoU, Precision, Recall, Specificity, Accuracy → `results/`
5. **Inference** (`scripts/02_apply_segmentation.py`) — load `.pth` from `models/` or GCS, run segmentation
6. **Feature extraction** (`scripts/03_extract_features.py`) — parallel pyradiomics on segmented images → GCS
7. **Classification** (`.venv/`, TBD) — train final ML/DL model on extracted features

### `src/mask/` Module

| File | Purpose |
|------|---------|
| `coco_annotations.py` | Convert HAM10000 binary masks → COCO JSON annotations |
| `coco_utils.py` | PyTorch `CocoDetection` dataset class + conversion utilities |
| `coco_eval.py` | `CocoEvaluator` — wraps pycocotools for bbox/segm evaluation |
| `engine.py` | `train_one_epoch()` and `evaluate()` training loops |
| `mask_evaluation.py` | Segmentation metrics: Dice, IoU, Precision, Recall, etc. |
| `transforms.py` | Data augmentation for detection/segmentation |
| `utils.py` | `MetricLogger`, `SmoothedValue`, distributed training helpers |

### `src/` Utilities

| File | Purpose |
|------|---------|
| `paths.py` | Portable path resolver — all dirs via `SCD_*` env vars |
| `viz.py` | Shared plotting utilities (`setup_style()`, `save_fig()`) |

### Data

- `data/HAM10000/` — raw images and binary masks (gitignored, download from Kaggle)
- `data/HAM10000/metadata.csv` — HAM10000 dataset metadata
- `models/` — trained `.pth` checkpoints (gitignored, stored in GCS)
- `results/` — training history CSVs, evaluation metrics, EDA plots
- ISIC and UDEM images/masks live on GCS mounts (not in repo)

### Path Overrides

All critical paths can be overridden via environment variables (see `src/paths.py`):

```bash
export SCD_DATA_DIR=/path/to/data
export SCD_MODELS_DIR=/path/to/models
export SCD_ISIC_IMAGES_DIR=/mnt/gcs/images
export SCD_ISIC_MASKS_DIR=/mnt/gcs/masks
```
