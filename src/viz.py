"""Shared plotting style and figure-saving helper.

Use `setup_style()` once at the top of a notebook, and `save_fig(fig, dir, name)`
instead of `plt.savefig(...)` so every figure lands in `results/...` with the
same dpi and tight bounding box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

CLASS_COLORS = {
    "Benign-melanocytic":        "#2196F3",
    "Benign-non-melanocytic":    "#4CAF50",
    "Malignant-melanocytic":     "#FF5722",
    "Malignant-non-melanocytic": "#9C27B0",
}

# HAM10000 7-class palette (dx codes)
HAM_CLASS_COLORS = {
    "akiec": "#9C27B0",
    "bcc":   "#FF5722",
    "bkl":   "#4CAF50",
    "df":    "#795548",
    "mel":   "#F44336",
    "nv":    "#2196F3",
    "vasc":  "#FFC107",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update({
        "figure.figsize":     (10, 6),
        "savefig.dpi":        300,
        "axes.titlesize":     15,
        "axes.titleweight":   "bold",
        "axes.labelsize":     13,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "legend.framealpha":  0.85,
        "figure.titlesize":   16,
        "figure.titleweight": "bold",
    })


def save_fig(fig, out_dir, name: str, *, dpi: int = 300) -> Path:
    """Save a Matplotlib figure under out_dir/name and return the resolved path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        name = f"{name}.png"
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
