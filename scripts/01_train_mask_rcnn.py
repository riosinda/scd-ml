"""Train Mask R-CNN on HAM10000 lesion segmentation.

Runs end-to-end on the local HAM10000 dataset:
    - splits 70/10/20 train/val/test (seeded)
    - fine-tunes maskrcnn_resnet50_fpn pretrained on COCO
    - early stopping on train loss
    - evaluates on the test set
    - saves model, training history CSV, test metrics CSV, and all plots

Usage (from repo root, with the .venv-mask environment active):
    python scripts/train_mask_rcnn.py

Run inside `tmux` so the SSH session doesn't take down the training:
    tmux new -s train
    source .venv-mask/bin/activate
    python scripts/train_mask_rcnn.py 2>&1 | tee train.log
    # Ctrl+B then D to detach

Outputs:
    models/{best_model.pth, maskrcnn_ham10000.pth}
    results/mask_rcnn/training/{training_history_*.csv, train_loss.png,
        val_dice_iou.png, val_aux_metrics.png, lr_schedule.png,
        val_metrics_panel.png}
    results/mask_rcnn/evaluation/{test_metrics_*.csv,
        test_mean_metrics_*.csv, test_metrics_barplot.png}
    results/mask_rcnn/samples/pred_overlay_4panel.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `from src.* import …` resolve when called from repo root or elsewhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt
import seaborn as sns

import src.mask.transforms as T
import src.mask.utils as utils
from src.mask.engine import train_one_epoch
from src.mask.mask_evaluation import evaluate
from src.paths import (
    HAM10000_DIR, MODELS_DIR,
    MASK_TRAINING_DIR, MASK_EVALUATION_DIR, MASK_SAMPLES_DIR,
)
from src.viz import save_fig, setup_style


# ─────────────── Configuration ──────────────────────────────────────────────
NUM_CLASSES  = 2          # background + lesion
BATCH_SIZE   = 8
NUM_WORKERS  = 0
NUM_EPOCHS   = 50
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE    = 15
GAMMA        = 0.1
PATIENCE     = 8
SEED         = 42


# ─────────────── Dataset ────────────────────────────────────────────────────
class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs  = sorted(os.listdir(os.path.join(root, "images")))
        self.masks = sorted(os.listdir(os.path.join(root, "masks")))

    def __getitem__(self, idx):
        img  = Image.open(os.path.join(self.root, "images", self.imgs[idx])).convert("RGB")
        mask = np.array(Image.open(os.path.join(self.root, "masks", self.masks[idx])))

        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            boxes.append([pos[1].min(), pos[0].min(), pos[1].max(), pos[0].max()])

        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            "boxes":    boxes,
            "labels":   torch.ones((num_objs,), dtype=torch.int64),
            "masks":    torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area":     (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd":  torch.zeros((num_objs,), dtype=torch.int64),
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ToTensor(torch.nn.Module):
    def forward(self, image, target):
        return F.to_tensor(image), target


def get_transform():
    return T.Compose([ToTensor()])


def filter_dataset(ds):
    keep = [i for i in range(len(ds)) if ds[i][1]["masks"].shape[0] > 0]
    return torch.utils.data.Subset(ds, keep)


# ─────────────── Model ──────────────────────────────────────────────────────
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ─────────────── Early stopping ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience, path, delta=0.0):
        self.patience = patience
        self.path = path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            if self.best_score is not None:
                print(f"  ⤵ val_loss {self.val_loss_min:.6f} → {val_loss:.6f}  (saving)")
            self.best_score = score
            self.val_loss_min = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ─────────────── Pipeline steps ─────────────────────────────────────────────
def prepare_loaders():
    raw = HAM10000Dataset(HAM10000_DIR, get_transform())
    ds  = filter_dataset(raw)

    total = len(ds)
    train_len = int(0.7 * total)
    temp_len  = total - train_len
    val_len   = int(0.3333 * temp_len)
    test_len  = temp_len - val_len

    g = torch.Generator().manual_seed(SEED)
    train_ds, temp_ds = random_split(ds, [train_len, temp_len], generator=g)
    g = torch.Generator().manual_seed(SEED)
    val_ds,   test_ds = random_split(temp_ds, [val_len, test_len], generator=g)
    print(f"Train: {train_len:,}  Val: {val_len:,}  Test: {test_len:,}")

    def loader(d, shuf):
        return torch.utils.data.DataLoader(
            d, batch_size=BATCH_SIZE, shuffle=shuf,
            num_workers=NUM_WORKERS, collate_fn=utils.collate_fn,
        )
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False), test_ds


def train(model, optimizer, scheduler, train_loader, val_loader, device):
    history = {k: [] for k in (
        "epochs", "train_loss", "lr",
        "val_dice", "val_iou", "val_precision",
        "val_recall", "val_specificity", "val_accuracy",
    )}
    best_path  = MODELS_DIR / "best_model.pth"
    early_stop = EarlyStopping(patience=PATIENCE, path=best_path)

    for epoch in range(NUM_EPOCHS):
        avg = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
        train_loss = avg.meters["loss"].global_avg
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        summary, _ = evaluate(model, val_loader, device)
        m = dict(zip(summary["metric"], summary["value"]))

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        for k in ("dice", "iou", "precision", "recall", "specificity", "accuracy"):
            history[f"val_{k}"].append(m[f"mean_{k}"])

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  loss={train_loss:.4f}  "
              f"dice={m['mean_dice']:.4f}  iou={m['mean_iou']:.4f}")

        early_stop(train_loss, model)
        if early_stop.early_stop:
            print("Early stopping triggered. Loading best checkpoint.")
            model.load_state_dict(torch.load(best_path))
            break
    return history


def save_training_plots(history):
    e = history["epochs"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=e, y=history["train_loss"], label="Train Loss", ax=ax)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Mask R-CNN — Train loss")
    ax.grid(True); ax.legend()
    save_fig(fig, MASK_TRAINING_DIR, "train_loss"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=e, y=history["val_dice"], label="Val Dice", ax=ax)
    sns.lineplot(x=e, y=history["val_iou"],  label="Val IoU",  ax=ax)
    ax.set(xlabel="Epoch", ylabel="Score", title="Mask R-CNN — Dice & IoU (validation)")
    ax.set_ylim(0.5, 1); ax.grid(True); ax.legend()
    save_fig(fig, MASK_TRAINING_DIR, "val_dice_iou"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, key in [
        ("Val Precision",   "val_precision"),
        ("Val Recall",      "val_recall"),
        ("Val Specificity", "val_specificity"),
        ("Val Accuracy",    "val_accuracy"),
    ]:
        sns.lineplot(x=e, y=history[key], label=label, ax=ax)
    ax.set(xlabel="Epoch", ylabel="Score", title="Mask R-CNN — Validation metrics")
    ax.set_ylim(0, 1); ax.grid(True); ax.legend()
    save_fig(fig, MASK_TRAINING_DIR, "val_aux_metrics"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=e, y=history["lr"], label="Learning Rate", ax=ax)
    ax.set(xlabel="Epoch", ylabel="LR", title="Learning rate schedule")
    ax.grid(True); ax.legend()
    save_fig(fig, MASK_TRAINING_DIR, "lr_schedule"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("Dice (Val)",      "val_dice",      None),
        ("IoU (Val)",       "val_iou",       "orange"),
        ("Precision (Val)", "val_precision", "green"),
        ("Recall (Val)",    "val_recall",    "red"),
    ]
    for ax, (label, key, color) in zip(axes.flat, panels):
        sns.lineplot(x=e, y=history[key], label=label, color=color, ax=ax)
        ax.set_title(label); ax.set_ylim(0.5, 1); ax.grid(True)
    fig.suptitle("Mask R-CNN — Validation metrics (panel)", y=1.01)
    plt.tight_layout()
    save_fig(fig, MASK_TRAINING_DIR, "val_metrics_panel"); plt.close(fig)


def save_test_plots(test_ds, model, device, all_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=all_results, x="metric", y="value", hue="metric",
        errorbar="sd", legend=False, ax=ax,
    )
    ax.set(title="Mask R-CNN — Métricas de evaluación (Test)",
           xlabel="Métrica", ylabel="Score")
    plt.tight_layout()
    save_fig(fig, MASK_EVALUATION_DIR, "test_metrics_barplot"); plt.close(fig)

    model.eval()
    img, _ = test_ds[0]
    with torch.no_grad():
        prediction = model([img.to(device)])
    masks  = prediction[0]["masks"]
    scores = prediction[0]["scores"]
    if len(masks) == 0 or scores[0] <= 0.5:
        print("  (no high-confidence prediction for first test image; skipping overlay)")
        return

    img_show    = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    mask_show   = masks[0, 0].mul(255).byte().cpu().numpy()
    binary_mask = mask_show > 128

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    axes[0, 0].imshow(img_show); axes[0, 0].set_title("Original Image", fontweight="bold"); axes[0, 0].axis("off")
    im = axes[0, 1].imshow(mask_show, cmap="magma"); axes[0, 1].set_title("Predicted Probability Map", fontweight="bold"); axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[1, 0].imshow(img_show); axes[1, 0].imshow(mask_show, alpha=0.5, cmap="jet")
    axes[1, 0].set_title("Overlayed Prediction", fontweight="bold"); axes[1, 0].axis("off")
    axes[1, 1].imshow(binary_mask, cmap="gray"); axes[1, 1].set_title("Binary Mask (>0.5)", fontweight="bold"); axes[1, 1].axis("off")
    plt.tight_layout(); plt.subplots_adjust(top=0.94)
    save_fig(fig, MASK_SAMPLES_DIR, "pred_overlay_4panel"); plt.close(fig)


# ─────────────── Main ───────────────────────────────────────────────────────
def main():
    setup_style()
    for d in (MODELS_DIR, MASK_TRAINING_DIR, MASK_EVALUATION_DIR, MASK_SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print("\n=== Preparing data ===")
    train_loader, val_loader, test_loader, test_ds = prepare_loaders()

    print("\n=== Building model ===")
    model = get_model(NUM_CLASSES).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print(f"\n=== Training {NUM_EPOCHS} epochs ===")
    history = train(model, optimizer, scheduler, train_loader, val_loader, device)

    print("\n=== Saving final model ===")
    final_path = MODELS_DIR / "maskrcnn_ham10000.pth"
    torch.save(model.state_dict(), final_path)
    print(f"  → {final_path}")

    print("\n=== Saving training history ===")
    history_df = pd.DataFrame({
        "epochs":          history["epochs"],
        "train_loss":      history["train_loss"],
        "val_dice":        history["val_dice"],
        "val_iou":         history["val_iou"],
        "val_precision":   history["val_precision"],
        "val_recall":      history["val_recall"],
        "val_specificity": history["val_specificity"],
        "val_accuracy":    history["val_accuracy"],
        "lr_history":      history["lr"],
    })
    history_csv = MASK_TRAINING_DIR / "training_history_maskrcnn_ham10000.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"  → {history_csv}")

    print("\n=== Generating training plots ===")
    save_training_plots(history)

    print("\n=== Evaluating on test set ===")
    mean_metrics, all_results = evaluate(model, test_loader, device)
    print(mean_metrics)
    test_csv = MASK_EVALUATION_DIR / "test_metrics_maskrcnn_ham10000.csv"
    mean_csv = MASK_EVALUATION_DIR / "test_mean_metrics_maskrcnn_ham10000.csv"
    all_results.to_csv(test_csv, index=False)
    mean_metrics.to_csv(mean_csv, index=False)
    print(f"  → {test_csv}")
    print(f"  → {mean_csv}")

    print("\n=== Generating test plots ===")
    save_test_plots(test_ds, model, device, all_results)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
