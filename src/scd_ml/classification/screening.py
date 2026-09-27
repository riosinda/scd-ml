"""Stage 1: screen channel sets, selectors and balancing with cheap default models."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from .pipeline import ModelConfig

SUMMARY_GROUPS = ["config", "channel_set", "selection", "k", "balancing", "model"]


def screening_configs(
    settings: dict[str, Any], *, seed: int, preprocessing: dict[str, float]
) -> list[ModelConfig]:
    """Every screening configuration, grouped by channel set so heads are reused."""
    return [
        ModelConfig(
            channel_set=channel_set,
            use_metadata=False,
            selection=selection,
            balancing=balancing,
            model=model,
            k=int(settings["k"]) if selection != "none" else None,
            seed=seed,
            preprocessing=dict(preprocessing),
        )
        for channel_set in settings["channel_sets"]
        for selection in settings["selections"]
        for balancing in settings["balancings"]
        for model in settings["models"]
    ]


def rank_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank (selection, balancing) pairs by their mean rank over channel x model cells.

    Averaging ranks across cells, instead of taking the single best cell, reduces the
    selection bias of picking a maximum among many noisy estimates.
    """
    ranked = summary.copy()
    ranked["cell_rank"] = ranked.groupby(["channel_set", "model"])["f1_macro_mean"].rank(
        ascending=False, method="average"
    )
    strategies = (
        ranked.groupby(["selection", "balancing"], dropna=False)
        .agg(
            mean_rank=("cell_rank", "mean"),
            mean_f1_macro=("f1_macro_mean", "mean"),
            n_cells=("cell_rank", "size"),
        )
        .reset_index()
    )
    complete = strategies["n_cells"].eq(strategies["n_cells"].max())
    strategies["complete"] = complete
    return strategies.sort_values(
        ["complete", "mean_rank", "mean_f1_macro"], ascending=[False, True, False]
    ).reset_index(drop=True)


def feature_stability(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selection frequency and pairwise fold Jaccard per (channel set, selector).

    ``selected`` has one row per (config, channel_set, selection, fold, feature). The
    selector precedes the balancer and the model, so any configuration sharing
    (channel set, selection) yields the same subsets; the first one is used.
    """
    keys = ["channel_set", "selection"]
    frequency_rows = []
    jaccard_rows = []
    for (channel_set, selection), rows in selected.groupby(keys, sort=True):
        rows = rows[rows["config"].eq(rows["config"].iloc[0])]
        by_fold = {
            int(fold): set(fold_rows["feature"]) for fold, fold_rows in rows.groupby("fold")
        }
        counts = rows.groupby("feature")["fold"].nunique()
        for feature, count in counts.items():
            frequency_rows.append(
                {
                    "channel_set": channel_set,
                    "selection": selection,
                    "feature": feature,
                    "n_folds_selected": int(count),
                    "selection_frequency": count / len(by_fold),
                }
            )
        for left, right in combinations(sorted(by_fold), 2):
            union = by_fold[left] | by_fold[right]
            jaccard_rows.append(
                {
                    "channel_set": channel_set,
                    "selection": selection,
                    "fold_a": left,
                    "fold_b": right,
                    "jaccard": len(by_fold[left] & by_fold[right]) / len(union),
                }
            )
    frequency = pd.DataFrame(
        frequency_rows,
        columns=[
            "channel_set",
            "selection",
            "feature",
            "n_folds_selected",
            "selection_frequency",
        ],
    ).sort_values(
        ["channel_set", "selection", "n_folds_selected", "feature"],
        ascending=[True, True, False, True],
    )
    jaccard = pd.DataFrame(
        jaccard_rows, columns=["channel_set", "selection", "fold_a", "fold_b", "jaccard"]
    )
    return frequency.reset_index(drop=True), jaccard
