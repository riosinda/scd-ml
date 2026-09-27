"""Grouped cross-validation over the locked development folds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from .columns import CLASS_ORDER, encode_target, feature_columns
from .datasets import split_development_fold
from .metrics import classification_metrics
from .pipeline import ModelConfig, build_head, fit_tail, selected_feature_names

PROBA_COLUMNS = [f"proba__{name}" for name in CLASS_ORDER]


@dataclass(frozen=True)
class FoldData:
    """Head-transformed matrices for one fold; the head saw only ``X_train`` rows."""

    fold: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    valid_image_ids: np.ndarray
    head: Pipeline


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    metrics: dict[str, float]
    selected_features: list[str]
    oof: pd.DataFrame


class FoldCache:
    """Fit the configuration-independent head once per (channels, metadata, fold)."""

    def __init__(
        self,
        cohort: pd.DataFrame,
        *,
        preprocessing: dict[str, float],
        folds: Iterable[int] = range(5),
    ) -> None:
        self.cohort = cohort
        self.preprocessing = dict(preprocessing)
        self.folds = tuple(int(fold) for fold in folds)
        self._cache: dict[tuple[str, bool], list[FoldData]] = {}

    def get(self, channel_set: str, use_metadata: bool) -> list[FoldData]:
        key = (channel_set, bool(use_metadata))
        if key not in self._cache:
            self._cache[key] = [
                self._fit_fold(fold, channel_set, use_metadata) for fold in self.folds
            ]
        return self._cache[key]

    def _fit_fold(self, fold: int, channel_set: str, use_metadata: bool) -> FoldData:
        training, validation = split_development_fold(self.cohort, fold)
        radiomics, metadata = feature_columns(
            self.cohort.columns, channel_set, use_metadata=use_metadata
        )
        columns = [*radiomics, *metadata]
        head = build_head(radiomics, metadata, self.preprocessing)
        return FoldData(
            fold=fold,
            X_train=head.fit_transform(training[columns]),
            y_train=encode_target(training["target"]),
            X_valid=head.transform(validation[columns]),
            y_valid=encode_target(validation["target"]),
            valid_image_ids=validation["image_id"].astype(str).to_numpy(),
            head=head,
        )


def evaluate_fold(config: ModelConfig, data: FoldData, *, n_jobs: int = 1) -> FoldResult:
    tail = fit_tail(config, data.X_train, data.y_train, n_jobs=n_jobs)
    proba = tail.predict_proba(data.X_valid)
    oof = pd.DataFrame(proba, columns=PROBA_COLUMNS)
    oof.insert(0, "y_true", data.y_valid)
    oof.insert(0, "fold", data.fold)
    oof.insert(0, "image_id", data.valid_image_ids)
    return FoldResult(
        fold=data.fold,
        n_train=len(data.y_train),
        n_valid=len(data.y_valid),
        metrics=classification_metrics(data.y_valid, proba),
        selected_features=selected_feature_names(tail, data.X_train.columns),
        oof=oof,
    )


def evaluate_config(
    config: ModelConfig,
    folds: Sequence[FoldData],
    *,
    fold_jobs: int = 1,
    model_jobs: int = 1,
) -> list[FoldResult]:
    """Evaluate ``config`` on every cached fold, optionally with folds in parallel."""
    if fold_jobs == 1:
        return [evaluate_fold(config, data, n_jobs=model_jobs) for data in folds]
    return Parallel(n_jobs=fold_jobs)(
        delayed(evaluate_fold)(config, data, n_jobs=model_jobs) for data in folds
    )


def fold_scores_frame(config: ModelConfig, results: Sequence[FoldResult]) -> pd.DataFrame:
    """One row per fold with the configuration factors and every metric."""
    rows = []
    for result in results:
        rows.append(
            {
                "config": config.name,
                "channel_set": config.channel_set,
                "use_metadata": config.use_metadata,
                "selection": config.selection,
                "k": config.k,
                "balancing": config.balancing,
                "model": config.model,
                "fold": result.fold,
                "n_train": result.n_train,
                "n_valid": result.n_valid,
                "n_selected_features": len(result.selected_features),
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_fold_scores(
    fold_scores: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    """Mean and sample standard deviation of every metric across folds."""
    metric_columns = [
        column
        for column in fold_scores.columns
        if column
        in {"f1_macro", "balanced_accuracy", "accuracy", "roc_auc_ovr_macro"}
        or column.startswith(("recall__", "f1__"))
    ]
    grouped = fold_scores.groupby(list(group_columns), dropna=False, sort=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_folds", grouped.size())
    return summary.reset_index()
