"""Stage-2 comparisons built from per-fold scores of the tuned studies.

``fold_scores`` has one row per (study, fold) with at least ``study``,
``channel_set``, ``use_metadata``, ``model``, ``fold``, ``n_train``, ``n_valid`` and
the metrics from :func:`scd_ml.classification.metrics.classification_metrics`.
"""

from __future__ import annotations

import pandas as pd

from .metrics import PRIMARY_METRIC
from .stats import corrected_paired_ttest, friedman_test


def compare_studies(fold_scores: pd.DataFrame, study_a: str, study_b: str) -> dict[str, float]:
    """Corrected paired t-test of ``study_a`` minus ``study_b`` on shared folds."""
    left = fold_scores[fold_scores["study"].eq(study_a)].set_index("fold")
    right = fold_scores[fold_scores["study"].eq(study_b)].set_index("fold")
    folds = sorted(set(left.index) & set(right.index))
    if len(folds) < 2:
        raise ValueError(f"{study_a} and {study_b} share fewer than two folds")
    return corrected_paired_ttest(
        left.loc[folds, PRIMARY_METRIC],
        right.loc[folds, PRIMARY_METRIC],
        n_train=float(left.loc[folds, "n_train"].mean()),
        n_test=float(left.loc[folds, "n_valid"].mean()),
    )


def study_means(fold_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_scores.groupby("study", sort=False)
    means = grouped.agg(
        channel_set=("channel_set", "first"),
        use_metadata=("use_metadata", "first"),
        model=("model", "first"),
        n_folds=("fold", "nunique"),
        f1_macro_mean=(PRIMARY_METRIC, "mean"),
        f1_macro_std=(PRIMARY_METRIC, "std"),
    )
    return means.reset_index()


def best_channel_set(fold_scores: pd.DataFrame) -> str:
    """Channel set with the highest F1-macro averaged over the radiomics-only models."""
    radiomics = study_means(fold_scores[~fold_scores["use_metadata"].astype(bool)])
    by_channel = radiomics.groupby("channel_set")["f1_macro_mean"].mean()
    return str(by_channel.idxmax())


def channel_ablation(fold_scores: pd.DataFrame, *, reference: str = "gray") -> pd.DataFrame:
    """Radiomics-only F1 per model and channel set, tested against ``reference``."""
    radiomics = fold_scores[~fold_scores["use_metadata"].astype(bool)]
    means = study_means(radiomics)
    rows = []
    for row in means.itertuples(index=False):
        baseline = means[means["model"].eq(row.model) & means["channel_set"].eq(reference)]
        result = {**row._asdict(), "reference": reference}
        if not baseline.empty and row.channel_set != reference:
            test = compare_studies(radiomics, row.study, baseline["study"].iloc[0])
            result.update({f"{key}_vs_reference": value for key, value in test.items()})
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["model", "channel_set"]).reset_index(drop=True)


def metadata_ablation(fold_scores: pd.DataFrame, channel_set: str) -> pd.DataFrame:
    """Per model: radiomics-only vs radiomics + clinical metadata on ``channel_set``."""
    subset = fold_scores[fold_scores["channel_set"].eq(channel_set)]
    means = study_means(subset)
    rows = []
    for model, group in means.groupby("model", sort=True):
        without = group[~group["use_metadata"].astype(bool)]
        with_metadata = group[group["use_metadata"].astype(bool)]
        if without.empty or with_metadata.empty:
            continue
        test = compare_studies(
            subset, with_metadata["study"].iloc[0], without["study"].iloc[0]
        )
        rows.append(
            {
                "model": model,
                "channel_set": channel_set,
                "f1_macro_radiomics": float(without["f1_macro_mean"].iloc[0]),
                "f1_macro_with_metadata": float(with_metadata["f1_macro_mean"].iloc[0]),
                **test,
            }
        )
    return pd.DataFrame(rows)


def pairwise_against(fold_scores: pd.DataFrame, winner: str) -> pd.DataFrame:
    rows = []
    for study in fold_scores["study"].drop_duplicates():
        if study == winner:
            continue
        test = compare_studies(fold_scores, winner, study)
        rows.append({"winner": winner, "other": study, **test})
    return pd.DataFrame(rows)


def friedman_over_models(fold_scores: pd.DataFrame, channel_set: str) -> dict[str, float] | None:
    subset = fold_scores[
        fold_scores["channel_set"].eq(channel_set) & ~fold_scores["use_metadata"].astype(bool)
    ]
    pivot = subset.pivot_table(index="fold", columns="study", values=PRIMARY_METRIC).dropna()
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return None
    return friedman_test({study: pivot[study].to_numpy() for study in pivot.columns})
