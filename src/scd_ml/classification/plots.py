"""Result figures for the classification stages.

Figures are built on ``matplotlib.figure.Figure`` objects instead of ``pyplot`` so
that importing this module never changes the global backend of a notebook or
script. Every public function saves PNG files and returns their paths.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from .columns import CHANNEL_SETS, CLASS_ORDER
from .metrics import confusion_frame

DPI = 300
# Same palette as ``src/viz.py`` so notebook and pipeline figures match.
CLASS_COLORS = {
    "Benign-melanocytic": "#2196F3",
    "Benign-non-melanocytic": "#4CAF50",
    "Malignant-melanocytic": "#FF5722",
    "Malignant-non-melanocytic": "#9C27B0",
}
SELECTION_ORDER = ("anova", "l1", "rfe")

def _styled[F: Callable[..., Path]](function: F) -> F:
    """Apply the seaborn style through ``rc_context`` so global rcParams stay untouched."""

    @functools.wraps(function)
    def wrapper(*args: object, **kwargs: object) -> Path:
        style = {**sns.axes_style("whitegrid"), **sns.plotting_context("notebook", 0.9)}
        with matplotlib.rc_context(style):
            return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _figure(width: float, height: float) -> tuple[Figure, object]:
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig, fig.add_subplot()


def _save(fig: Figure, directory: Path, name: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def feature_family(feature: str) -> str:
    """PyRadiomics class of ``<channel>__<imageType>_<class>_<name>``; metadata otherwise."""
    channel, separator, rest = str(feature).partition("__")
    known_channels = {name for channels in CHANNEL_SETS.values() for name in channels}
    if not separator or channel not in known_channels:
        return "metadata"
    parts = rest.split("_", 2)
    return parts[1] if len(parts) >= 2 else rest


# --------------------------------------------------------------------------- stage 1


@_styled
def plot_strategy_ranking(strategies: pd.DataFrame, directory: Path) -> Path:
    data = strategies.assign(
        strategy=strategies["selection"] + " / " + strategies["balancing"]
    ).sort_values("mean_rank")
    fig, ax = _figure(8, 0.45 * len(data) + 1.5)
    sns.barplot(data=data, x="mean_rank", y="strategy", color="#4C72B0", ax=ax)
    for patch, f1 in zip(ax.patches, data["mean_f1_macro"], strict=False):
        ax.annotate(
            f"F1 {f1:.3f}",
            (patch.get_width(), patch.get_y() + patch.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set(
        xlabel="Mean rank across channel × model cells (lower is better)",
        ylabel="Selection / balancing",
        title="Stage 1 — strategy ranking",
    )
    return _save(fig, directory, "strategy_ranking")


@_styled
def plot_screening_heatmap(summary: pd.DataFrame, directory: Path) -> Path:
    data = summary.assign(
        strategy=summary["selection"] + " / " + summary["balancing"],
        cell=summary["channel_set"] + " · " + summary["model"],
    )
    pivot = data.pivot_table(index="strategy", columns="cell", values="f1_macro_mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = _figure(1.3 * pivot.shape[1] + 3, 0.45 * pivot.shape[0] + 1.5)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "F1-macro"}, ax=ax)
    ax.grid(False)
    ax.set(xlabel="Channel set · model", ylabel="Selection / balancing")
    ax.set_title("Stage 1 — mean CV F1-macro per configuration")
    return _save(fig, directory, "screening_f1_heatmap")


@_styled
def plot_selection_jaccard(jaccard: pd.DataFrame, directory: Path) -> Path:
    fig, ax = _figure(7, 4.5)
    sns.barplot(
        data=jaccard,
        x="selection",
        y="jaccard",
        hue="channel_set",
        order=[s for s in SELECTION_ORDER if s in set(jaccard["selection"])],
        errorbar="sd",
        ax=ax,
    )
    ax.set(
        ylim=(0, 1),
        xlabel="Feature selector",
        ylabel="Jaccard between fold subsets",
        title="Feature-selection stability across folds",
    )
    ax.legend(title="Channel set")
    return _save(fig, directory, "feature_selection_stability")


@_styled
def plot_selected_features(
    frequency: pd.DataFrame, directory: Path, name: str, *, title: str, top: int = 20
) -> Path:
    """Per selector: top-``top`` features by fold frequency and their radiomics families.

    ``frequency`` has ``selection``, ``feature``, ``n_folds_selected`` and
    ``selection_frequency`` columns (one selector per row group).
    """
    selections = list(dict.fromkeys(frequency["selection"]))
    fig = Figure(figsize=(15, 0.28 * top * len(selections) + 1.5), layout="constrained")
    axes = fig.subplots(len(selections), 2, squeeze=False, width_ratios=[2.2, 1])
    for row, selection in enumerate(selections):
        rows = frequency[frequency["selection"].eq(selection)]
        ranked = rows.sort_values(["selection_frequency", "feature"], ascending=[False, True])
        sns.barplot(
            data=ranked.head(top),
            x="selection_frequency",
            y="feature",
            color="#4C72B0",
            ax=axes[row, 0],
        )
        axes[row, 0].set(
            xlim=(0, 1),
            xlabel="Fraction of folds selecting the feature",
            ylabel="",
            title=f"{selection}: top {min(top, len(ranked))} of {len(ranked)} features",
        )
        axes[row, 0].tick_params(axis="y", labelsize=7)
        families = (
            rows.assign(family=rows["feature"].map(feature_family))
            .groupby("family")["selection_frequency"]
            .sum()
            .sort_values(ascending=False)
        )
        sns.barplot(x=families.to_numpy(), y=families.index, color="#55A868", ax=axes[row, 1])
        axes[row, 1].set(
            xlabel="Selections per fold (sum of frequencies)",
            ylabel="",
            title=f"{selection}: radiomics family",
        )
    fig.suptitle(title)
    return _save(fig, directory, name)


def screening_figures(
    summary: pd.DataFrame,
    strategies: pd.DataFrame,
    directory: Path,
    *,
    frequency: pd.DataFrame | None = None,
    jaccard: pd.DataFrame | None = None,
) -> list[Path]:
    paths = [
        plot_strategy_ranking(strategies, directory),
        plot_screening_heatmap(summary, directory),
    ]
    if jaccard is not None and not jaccard.empty:
        paths.append(plot_selection_jaccard(jaccard, directory))
    if frequency is not None and not frequency.empty:
        for channel_set, rows in frequency.groupby("channel_set", sort=True):
            paths.append(
                plot_selected_features(
                    rows,
                    directory,
                    f"selected_features_{channel_set}",
                    title=f"Stage 1 — features selected on channel set '{channel_set}'",
                )
            )
    return paths


# --------------------------------------------------------------------------- stage 2


@_styled
def plot_studies(summary: pd.DataFrame, directory: Path) -> Path:
    data = summary.sort_values("f1_macro_mean", ascending=True).reset_index(drop=True)
    labels = [
        f"{row.study} (k={int(row.k)})" if pd.notna(row.k) else str(row.study)
        for row in data.itertuples()
    ]
    palette = dict(zip(sorted(data["model"].unique()), sns.color_palette("tab10"), strict=False))
    fig, ax = _figure(9, 0.35 * len(data) + 1.5)
    ax.barh(
        labels,
        data["f1_macro_mean"],
        xerr=data["f1_macro_std"],
        color=[palette[model] for model in data["model"]],
        hatch=["//" if meta else "" for meta in data["use_metadata"].astype(bool)],
        capsize=3,
    )
    handles = [Patch(color=color, label=model) for model, color in palette.items()]
    ax.legend(
        handles=handles,
        title="Model (hatched: + metadata)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    lower = max(0.0, float((data["f1_macro_mean"] - data["f1_macro_std"]).min()) - 0.05)
    ax.set(
        xlim=(lower, min(1.0, float(data["f1_macro_mean"].max()) + 0.08)),
        xlabel="CV F1-macro (mean ± sd over folds)",
        title="Stage 2 — tuned studies",
    )
    return _save(fig, directory, "studies_f1")


@_styled
def plot_channel_ablation(ablation: pd.DataFrame, directory: Path) -> Path:
    order = [c for c in CHANNEL_SETS if c in set(ablation["channel_set"])]
    fig, ax = _figure(9, 4.8)
    sns.barplot(
        data=ablation, x="model", y="f1_macro_mean", hue="channel_set", hue_order=order, ax=ax
    )
    ax.set(xlabel="Model", ylabel="CV F1-macro", title="Channel ablation (radiomics only)")
    if "p_value_vs_reference" in ablation:
        models = list(dict.fromkeys(ablation["model"]))
        for container, channel in zip(ax.containers, order, strict=False):
            for patch, model in zip(container, models, strict=False):
                row = ablation[ablation["model"].eq(model) & ablation["channel_set"].eq(channel)]
                p_value = row["p_value_vs_reference"].iloc[0] if not row.empty else np.nan
                if pd.notna(p_value) and p_value < 0.05:
                    ax.annotate(
                        "*",
                        (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                        ha="center",
                        va="bottom",
                    )
        reference = ablation["reference"].iloc[0]
        ax.text(
            0.01, 0.98, f"* p < 0.05 vs {reference} (Nadeau–Bengio)",
            transform=ax.transAxes, va="top", fontsize=8,
        )
    ax.legend(title="Channel set", loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, directory, "channel_ablation")


@_styled
def plot_metadata_ablation(ablation: pd.DataFrame, directory: Path) -> Path:
    data = ablation.sort_values("f1_macro_with_metadata")
    fig, ax = _figure(8, 0.5 * len(data) + 1.8)
    y = np.arange(len(data))
    ax.hlines(y, data["f1_macro_radiomics"], data["f1_macro_with_metadata"], color="grey")
    ax.scatter(data["f1_macro_radiomics"], y, label="Radiomics", zorder=3)
    ax.scatter(data["f1_macro_with_metadata"], y, label="Radiomics + metadata", zorder=3)
    for position, row in zip(y, data.itertuples(), strict=True):
        ax.annotate(
            f"p={row.p_value:.3f}",
            (max(row.f1_macro_radiomics, row.f1_macro_with_metadata), position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y, data["model"])
    ax.set(
        xlabel="CV F1-macro",
        title=f"Metadata ablation on channel set '{data['channel_set'].iloc[0]}'",
    )
    ax.legend(loc="lower right")
    return _save(fig, directory, "metadata_ablation")


@_styled
def plot_per_class_recall(summary: pd.DataFrame, directory: Path) -> Path:
    columns = [f"recall__{name}_mean" for name in CLASS_ORDER if f"recall__{name}_mean" in summary]
    data = summary.set_index("study")[columns]
    data.columns = [column.removeprefix("recall__").removesuffix("_mean") for column in columns]
    fig, ax = _figure(9, 0.35 * len(data) + 2)
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1,
        cbar_kws={"label": "Recall"}, ax=ax,
    )
    ax.grid(False)
    ax.set(xlabel="", ylabel="", title="Stage 2 — mean CV recall per class")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, directory, "per_class_recall")


@_styled
def plot_optimization_history(history: pd.DataFrame, directory: Path) -> Path:
    """Best-so-far objective per study; ``history`` has study, model, number, value."""
    models = list(dict.fromkeys(history["model"]))
    fig = Figure(figsize=(4.2 * len(models), 3.8), layout="constrained")
    axes = fig.subplots(1, len(models), squeeze=False, sharey=True)[0]
    for ax, model in zip(axes, models, strict=True):
        for study, rows in history[history["model"].eq(model)].groupby("study", sort=True):
            rows = rows.sort_values("number")
            ax.plot(rows["number"], rows["value"].cummax(), label=study)
        ax.set(title=model, xlabel="Trial")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Best CV F1-macro so far")
    fig.suptitle("Stage 2 — Optuna optimization history (completed trials)")
    return _save(fig, directory, "optimization_history")


def tuning_figures(
    summary: pd.DataFrame,
    directory: Path,
    *,
    channel_ablation: pd.DataFrame | None = None,
    metadata_ablation: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    winner_selection: pd.DataFrame | None = None,
    winner: str | None = None,
) -> list[Path]:
    paths = [plot_studies(summary, directory), plot_per_class_recall(summary, directory)]
    if channel_ablation is not None and not channel_ablation.empty:
        paths.append(plot_channel_ablation(channel_ablation, directory))
    if metadata_ablation is not None and not metadata_ablation.empty:
        paths.append(plot_metadata_ablation(metadata_ablation, directory))
    if history is not None and not history.empty:
        paths.append(plot_optimization_history(history, directory))
    if winner_selection is not None and not winner_selection.empty:
        paths.append(
            plot_selected_features(
                winner_selection,
                directory,
                "winner_selected_features",
                title=f"Stage 2 — features selected by the winner '{winner}' across folds",
            )
        )
    return paths


# --------------------------------------------------------------------------- stage 4


@_styled
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, directory: Path) -> Path:
    counts = confusion_frame(y_true, y_pred)
    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig = Figure(figsize=(15, 6), layout="constrained")
    left, right = fig.subplots(1, 2)
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False, ax=left)
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=right)
    left.set_title("Counts")
    right.set_title("Row-normalized (recall)")
    for ax in (left, right):
        ax.grid(False)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Frozen test — confusion matrix")
    return _save(fig, directory, "confusion_matrix")


@_styled
def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, directory: Path) -> Path:
    fig = Figure(figsize=(13, 5.5), layout="constrained")
    roc_ax, pr_ax = fig.subplots(1, 2)
    for index, name in enumerate(CLASS_ORDER):
        positive = (np.asarray(y_true) == index).astype(int)
        if positive.sum() == 0 or positive.sum() == len(positive):
            continue
        color = CLASS_COLORS[name]
        fpr, tpr, _ = roc_curve(positive, proba[:, index])
        roc_ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        precision, recall, _ = precision_recall_curve(positive, proba[:, index])
        ap = average_precision_score(positive, proba[:, index])
        pr_ax.plot(recall, precision, color=color, label=f"{name} (AP {ap:.3f})")
        pr_ax.axhline(positive.mean(), color=color, linestyle=":", linewidth=0.8)
    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (one-vs-rest)")
    pr_ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–recall (dotted: prevalence)",
        ylim=(0, 1.02),
    )
    roc_ax.legend(loc="lower right", fontsize=8)
    pr_ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Frozen test — discrimination per class")
    return _save(fig, directory, "roc_pr_curves")


@_styled
def plot_per_class_metrics(per_class: pd.DataFrame, directory: Path) -> Path:
    data = per_class.melt(
        id_vars=["class", "support"], value_vars=["recall", "f1"], var_name="metric"
    )
    fig, ax = _figure(9, 4.5)
    sns.barplot(data=data, x="class", y="value", hue="metric", ax=ax)
    labels = [
        f"{name}\n(n={int(n)})"
        for name, n in zip(per_class["class"], per_class["support"], strict=True)
    ]
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylim=(0, 1), xlabel="", ylabel="Score", title="Frozen test — per-class recall and F1")
    return _save(fig, directory, "per_class_metrics")


@_styled
def plot_final_feature_families(selected_features: Sequence[str], directory: Path) -> Path:
    families = pd.Series([feature_family(f) for f in selected_features]).value_counts()
    fig, ax = _figure(7, 0.4 * len(families) + 1.5)
    sns.barplot(x=families.to_numpy(), y=families.index, color="#55A868", ax=ax)
    ax.set(
        xlabel="Selected features",
        title=f"Final model — {len(selected_features)} selected features by family",
    )
    return _save(fig, directory, "final_selected_features")


def holdout_figures(
    y_true: np.ndarray,
    proba: np.ndarray,
    per_class: pd.DataFrame,
    directory: Path,
    *,
    selected_features: Sequence[str] | None = None,
) -> list[Path]:
    y_pred = np.asarray(proba).argmax(axis=1)
    paths = [
        plot_confusion(y_true, y_pred, directory),
        plot_roc_pr(y_true, np.asarray(proba), directory),
        plot_per_class_metrics(per_class, directory),
    ]
    if selected_features:
        paths.append(plot_final_feature_families(selected_features, directory))
    return paths
