"""Four-class classification metrics; F1-macro is the primary criterion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .columns import CLASS_ORDER

PRIMARY_METRIC = "f1_macro"


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Summary metrics from integer labels and ``CLASS_ORDER``-aligned probabilities."""
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(y_true), len(CLASS_ORDER)):
        raise ValueError(f"probabilities must have shape (n, {len(CLASS_ORDER)})")
    labels = np.arange(len(CLASS_ORDER))
    y_pred = proba.argmax(axis=1)
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc_ovr_macro": roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=labels
        ),
    }
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, recall, f1 in zip(CLASS_ORDER, recalls, f1s, strict=True):
        metrics[f"recall__{name}"] = recall
        metrics[f"f1__{name}"] = f1
    return {key: float(value) for key, value in metrics.items()}


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Confusion matrix with true classes as rows and predicted classes as columns."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_ORDER)))
    return pd.DataFrame(
        matrix,
        index=pd.Index(CLASS_ORDER, name="true"),
        columns=pd.Index(CLASS_ORDER, name="predicted"),
    )


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = np.arange(len(CLASS_ORDER))
    return pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "support": np.bincount(y_true, minlength=len(CLASS_ORDER)),
            "recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            "f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        }
    )
