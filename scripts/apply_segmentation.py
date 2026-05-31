"""Apply the trained Mask R-CNN to the ISIC archive and save binary masks.

Runs end-to-end:
    - loads maskrcnn_ham10000.pth from MODELS_DIR
    - segments every image in ISIC_IMAGES_DIR
    - saves a binary mask per image to ISIC_MASKS_DIR (same filename)
    - renders a 3x3 visual validation grid to SEGMENTATION_DIR

Usage (from repo root, with the .venv-mask environment active):
    python scripts/apply_segmentation.py

Run inside `tmux` so the SSH session doesn't take down the inference:
    tmux new -s segment
    source .venv-mask/bin/activate
    python scripts/apply_segmentation.py 2>&1 | tee segment.log
    # Ctrl+B then D to detach

Outputs:
    {ISIC_MASKS_DIR}/<image_name>     one binary PNG/JPG mask per input image
    {SEGMENTATION_DIR}/sample_predictions.png
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.paths import (
    ISIC_IMAGES_DIR, ISIC_MASKS_DIR, MODELS_DIR, SEGMENTATION_DIR,
)
from src.viz import save_fig, setup_style


# ─────────────── Configuration ──────────────────────────────────────────────
NUM_CLASSES    = 2          # background + lesion
MASK_THRESHOLD = 0.5
IMG_EXTS       = (".jpg", ".jpeg", ".png")
SAMPLE_SEED    = 0
SAMPLE_N       = 9


# ─────────────── Model ──────────────────────────────────────────────────────
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ─────────────── Pipeline steps ─────────────────────────────────────────────
def load_model(device):
    model_path = MODELS_DIR / "maskrcnn_ham10000.pth"
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {model_path}")
    return model


def segment_all(model, device):
    img_names = [f for f in os.listdir(ISIC_IMAGES_DIR) if f.lower().endswith(IMG_EXTS)]
    print(f"Images on disk: {len(img_names):,}")

    for img_name in tqdm(img_names, desc="Processing images"):
        image = Image.open(ISIC_IMAGES_DIR / img_name).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = model([image_tensor])

        if len(prediction[0]["masks"]) > 0:
            raw_mask = prediction[0]["masks"][0, 0]
            binary_mask = raw_mask > MASK_THRESHOLD
            mask_arr = binary_mask.mul(255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask_arr).convert("L")
        else:
            mask_image = Image.new("L", image.size, 0)

        # Same filename as input — downstream code in notebook 03 expects this.
        mask_image.save(ISIC_MASKS_DIR / img_name)


def save_validation_grid():
    mask_files = [f for f in os.listdir(ISIC_MASKS_DIR) if f.lower().endswith(IMG_EXTS)]
    if not mask_files:
        print("  (no masks found; skipping validation grid)")
        return

    random.seed(SAMPLE_SEED)
    sample = random.sample(mask_files, min(SAMPLE_N, len(mask_files)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("ISIC — Predicted segmentation samples (n=9)", y=0.92)

    for ax, fname in zip(axes.flat, sample):
        img = np.array(Image.open(ISIC_IMAGES_DIR / fname).convert("RGB"))
        mask = np.array(Image.open(ISIC_MASKS_DIR / fname).convert("L"))
        ax.imshow(img)
        ax.imshow(mask, alpha=0.45, cmap="Reds")
        ax.set_title(fname, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, SEGMENTATION_DIR, "sample_predictions")
    plt.close(fig)


# ─────────────── Main ───────────────────────────────────────────────────────
def main():
    setup_style()
    for d in (ISIC_MASKS_DIR, SEGMENTATION_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print(f"Source images: {ISIC_IMAGES_DIR}")
    print(f"Output masks:  {ISIC_MASKS_DIR}")

    print("\n=== Loading model ===")
    model = load_model(device)

    print("\n=== Segmenting ISIC archive ===")
    segment_all(model, device)

    print("\n=== Generating validation grid ===")
    save_validation_grid()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
