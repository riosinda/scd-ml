# Skin Cancer Detection — ML Thesis

Skin cancer classification pipeline using lesion segmentation (Mask R-CNN), radiomic feature extraction, and ML/DL classification. Datasets: HAM10000, ISIC Archive, UDEM.

---

## Pipeline

```
EDA  →  Mask R-CNN training  →  Inference (segmentation)  →  Feature extraction  →  ML/DL model
(HAM10000 + ISIC + UDEM)       (HAM10000)                    (ISIC on GCS)          (radiomic features)
```

---

## Repository Structure

```
scd-ml/
├── notebooks/              # EDA notebooks (Jupyter)
│   └── archive/            # Deprecated notebooks (logic moved to scripts/)
├── scripts/                # Training, inference, and feature extraction scripts
├── src/
│   ├── paths.py            # Portable path resolver (SCD_* env var overrides)
│   ├── viz.py              # Shared plotting utilities
│   └── mask/               # Mask R-CNN source (torchvision-based)
├── requirements/
│   ├── mask.txt            # Python 3.12.6 — Mask R-CNN
│   ├── features.txt        # Python 3.7 — radiomic extraction
│   └── ml.txt              # Latest Python — final classification model
├── results/                # Plots, CSVs, model evaluation outputs
├── models/                 # Trained .pth checkpoints (gitignored, store in GCS)
└── data/                   # Raw datasets (gitignored, see Datasets section)
```

---

## Datasets

| Dataset | Location | Notes |
|---------|----------|-------|
| HAM10000 | `data/HAM10000/` (local) | Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection). Includes `images/`, `masks/`, `metadata.csv`. |
| ISIC Archive | GCS bucket (mount) | Images and metadata — mounted at `~/data/gcs/` via gcsfuse |
| UDEM | GCS bucket (mount) | Custom dataset — mounted at a separate GCS path |

Expected local structure for HAM10000:
```
data/
└── HAM10000/
    ├── images/         # JPEG lesion images
    ├── masks/          # Binary PNG masks
    └── metadata.csv    # Ground-truth labels and demographics
```

---

## Python Environments

Three separate environments managed with **pyenv**. All are gitignored.

| Environment | Python | Purpose |
|-------------|--------|---------|
| `.venv-mask/` | 3.12.6 | Mask R-CNN training and inference |
| `.venv-features/` | 3.7 | Radiomic feature extraction (pyradiomics) |
| `.venv/` | latest | Final ML/DL classification model |

Set up each environment:
```bash
# Install pyenv: https://github.com/pyenv/pyenv
pyenv install 3.12.6
pyenv install 3.7.17

# Mask R-CNN environment
pyenv local 3.12.6
python -m venv .venv-mask
source .venv-mask/bin/activate
pip install -r requirements/mask.txt

# Feature extraction environment (Python 3.7 — order matters)
pyenv local 3.7.17
python -m venv .venv-features
source .venv-features/bin/activate
pip install numpy pandas opencv-python pydicom
pip install SimpleITK --only-binary :all:
pip install pyradiomics

# ML/DL model environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/ml.txt
```

> **Note:** `requirements/` files are placeholders. Run `pip freeze > requirements/<env>.txt` on the GCP VM after installing dependencies.

---

## GCP Setup

### Instance Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | — | L4 |
| vCPUs | 16 | 32 |
| RAM | 128 GB | 128 GB |
| Boot disk | 100 GB SSD | 200 GB SSD |
| OS | Debian 11 / Ubuntu 22.04 | — |

Mask R-CNN training requires at least 128 GB of RAM due to the size of HAM10000 loaded in memory.

### GCS Storage Layout

All large assets (images, masks, models, feature CSVs) live in a GCS bucket to avoid local storage limits and enable sharing between machines.

```
gs://your-bucket/scd-ml/
├── images/           # ISIC archive images (flat, ~470 K JPEGs)
├── masks/            # Segmentation masks output by apply_segmentation.py
├── models/           # Trained .pth checkpoints
└── features/         # Extracted radiomic feature CSVs (500 MB+)
```

### Mount ISIC Images with gcsfuse

```bash
# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-$(lsb_release -cs)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y gcsfuse

# Authenticate
gcloud auth application-default login

# Mount buckets
mkdir -p ~/data/gcs ~/data/gcs-masks
gcsfuse --implicit-dirs your-bucket/images ~/data/gcs
gcsfuse --implicit-dirs your-bucket/masks  ~/data/gcs-masks
```

### Environment Variables

`src/paths.py` reads these to resolve all project paths:

```bash
export SCD_DATA_DIR=/path/to/data            # default: repo/data/
export SCD_MODELS_DIR=/path/to/models        # default: repo/models/
export SCD_RESULTS_DIR=/path/to/results      # default: repo/results/
export SCD_HAM10000_DIR=/path/to/HAM10000    # default: $SCD_DATA_DIR/HAM10000
export SCD_ISIC_DIR=~/data/gcs               # default: ~/data/gcs
export SCD_ISIC_IMAGES_DIR=~/data/gcs        # default: $SCD_ISIC_DIR
export SCD_ISIC_MASKS_DIR=~/data/gcs-masks   # default: ~/data/gcs-masks
export SCD_GCS_BUCKET=gs://your-bucket/scd-ml
```

Add these to `~/.bashrc` or `~/.zshrc` on the GCP instance.

### Sync Feature CSVs

```bash
export SCD_GCS_BUCKET=gs://your-bucket/scd-ml

# After feature extraction — upload to GCS
gsutil -m rsync -r -x '\.gitkeep$' results/features/ $SCD_GCS_BUCKET/features/

# On local machine — download for analysis
gsutil -m rsync -r $SCD_GCS_BUCKET/features/ results/features/
```

---

## EDA Notebooks

Launch with:
```bash
jupyter notebook notebooks/eda/
```

| Notebook | Dataset | Description |
|----------|---------|-------------|
| `notebooks/eda/01_isic_archive.ipynb` | ISIC Archive | Class distribution, demographics, image dimensions |
| `notebooks/eda/02_ham10000.ipynb` | HAM10000 | Class balance, demographics, mask quality |
| `notebooks/eda/03_udem.ipynb` | UDEM | Custom dataset exploration |

---

## Scripts (Training & Inference)

Scripts are numbered by pipeline stage. Run them in a **tmux session** on GCP to survive SSH disconnections:

```bash
tmux new -s training
source .venv-mask/bin/activate
python scripts/01_train_mask_rcnn.py
```

| Script | Environment | Description |
|--------|-------------|-------------|
| `scripts/01_train_mask_rcnn.py` | `.venv-mask` | Train Mask R-CNN on HAM10000 |
| `scripts/02_apply_segmentation.py` | `.venv-mask` | Inference on ISIC images → save masks to GCS |
| `scripts/03_extract_features.py` | `.venv-features` | Parallel radiomic extraction (resumable) |

---

## Results

Generated artifacts are saved to `results/`:

```
results/
├── eda/
│   ├── ham10000/       # Class distribution, demographics plots
│   └── isic/           # Class distribution, age, sex, anatomical site plots
├── mask_rcnn/
│   ├── training/       # Loss curves, LR schedule, training history CSV
│   ├── evaluation/     # Dice, IoU, Precision, Recall per image (CSV + barplot)
│   └── samples/        # Prediction overlay samples
├── processed/          # Intermediate parquet/CSV artifacts
└── features/           # Radiomic feature CSVs (gitignored — synced via GCS)
```

Trained model checkpoints (`*.pth`) are gitignored — store and retrieve them via GCS.
