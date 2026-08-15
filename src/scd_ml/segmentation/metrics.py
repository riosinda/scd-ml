"""Pixel-level lesion segmentation metrics with explicit empty-mask semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

METRIC_NAMES = ("dice", "iou", "precision", "recall", "specificity", "accuracy")


def _safe_ratio(numerator: int, denominator: int, *, zero_value: float) -> float:
    return float(numerator / denominator) if denominator else zero_value


def binary_segmentation_metrics(prediction: Any, target: Any) -> dict[str, float | int]:
    """Compute metrics without epsilon-based inflation.

    Precision and recall use ``zero_division=0``. Dice and IoU are 1 only when
    both masks are empty, because the two segmentations then match exactly.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {truth.shape}")

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    both_empty = not pred.any() and not truth.any()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn, zero_value=float(both_empty)),
        "iou": _safe_ratio(tp, tp + fp + fn, zero_value=float(both_empty)),
        "precision": _safe_ratio(tp, tp + fp, zero_value=0.0),
        "recall": _safe_ratio(tp, tp + fn, zero_value=0.0),
        "specificity": _safe_ratio(tn, tn + fp, zero_value=0.0),
        "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn, zero_value=0.0),
    }


def summarize_metrics(per_image: pd.DataFrame) -> pd.DataFrame:
    if per_image.empty:
        raise ValueError("Cannot summarize an empty evaluation")
    totals = per_image[["tp", "fp", "fn", "tn"]].sum()
    micro = binary_segmentation_metrics_from_counts(**totals.astype(int).to_dict())
    rows = []
    for metric in METRIC_NAMES:
        rows.append(
            {
                "metric": metric,
                "macro_mean": float(per_image[metric].mean()),
                "macro_std": float(per_image[metric].std(ddof=0)),
                "micro_value": float(micro[metric]),
            }
        )
    return pd.DataFrame(rows)


def binary_segmentation_metrics_from_counts(
    *, tp: int, fp: int, fn: int, tn: int
) -> dict[str, float]:
    both_empty = tp == fp == fn == 0
    return {
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn, zero_value=float(both_empty)),
        "iou": _safe_ratio(tp, tp + fp + fn, zero_value=float(both_empty)),
        "precision": _safe_ratio(tp, tp + fp, zero_value=0.0),
        "recall": _safe_ratio(tp, tp + fn, zero_value=0.0),
        "specificity": _safe_ratio(tn, tn + fp, zero_value=0.0),
        "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn, zero_value=0.0),
    }


def _source_id(target: Mapping[str, Any], fallback: int) -> str:
    if "source_id" in target:
        return str(target["source_id"])
    value = target.get("image_id", fallback)
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def _select_prediction(
    output: Mapping[str, Any],
    *,
    target_shape: tuple[int, int],
    score_threshold: float,
    mask_threshold: float,
):
    import torch
    import torch.nn.functional as functional

    scores = output.get("scores")
    masks = output.get("masks")
    if scores is None or masks is None or len(scores) == 0:
        output_device = masks.device if masks is not None else None
        return torch.zeros(target_shape, dtype=torch.bool, device=output_device), None, 0

    valid = torch.where(scores >= score_threshold)[0]
    if len(valid) == 0:
        return torch.zeros(target_shape, dtype=torch.bool, device=scores.device), None, 0
    best = valid[torch.argmax(scores[valid])]
    probability = masks[best, 0]
    if tuple(probability.shape) != target_shape:
        probability = functional.interpolate(
            probability[None, None], size=target_shape, mode="bilinear", align_corners=False
        )[0, 0]
    return probability >= mask_threshold, float(scores[best].item()), int(len(valid))


def evaluate_segmenter(
    model,
    data_loader,
    device,
    *,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model([image.to(device) for image in images])
            for position, (output, target) in enumerate(zip(outputs, targets, strict=True)):
                target_masks = target["masks"].to(device)
                shape = tuple(images[position].shape[-2:])
                if len(target_masks):
                    truth = target_masks.bool().any(dim=0)
                else:
                    truth = torch.zeros(shape, dtype=torch.bool, device=device)
                prediction, score, detections = _select_prediction(
                    output,
                    target_shape=shape,
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                )
                metrics = binary_segmentation_metrics(
                    prediction.detach().cpu().numpy(), truth.detach().cpu().numpy()
                )
                records.append(
                    {
                        "image_id": _source_id(target, len(records)),
                        "score": score,
                        "num_detections": detections,
                        "status": "detected" if detections else "no_detection",
                        **metrics,
                    }
                )
    per_image = pd.DataFrame.from_records(records)
    return summarize_metrics(per_image), per_image
