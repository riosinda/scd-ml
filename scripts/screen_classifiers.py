#!/usr/bin/env python3
"""Stage 1: screen channel sets x selection x balancing with LogReg and XGBoost.

Every configuration runs on the locked grouped development folds. Progress is
appended to ``fold_scores.csv`` and resumed by configuration name; only
``--overwrite`` discards it. Figures go to ``<output-dir>/figures`` and every
configuration is mirrored as a nested MLflow run unless ``--no-mlflow`` is passed.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from scd_ml.classification.cv import (
    FoldCache,
    evaluate_config,
    fold_scores_frame,
    summarize_fold_scores,
)
from scd_ml.classification.data import load_cohort
from scd_ml.classification.plots import screening_figures
from scd_ml.classification.runs import (
    add_cohort_arguments,
    add_tracking_arguments,
    cohort_inputs,
    load_yaml,
    parse_folds,
    prepare_run_dir,
    write_json,
)
from scd_ml.classification.screening import (
    SUMMARY_GROUPS,
    feature_stability,
    rank_strategies,
    screening_configs,
)
from scd_ml.classification.tracking import Tracker, log_cv_child, numeric_metrics
from scd_ml.paths import CLASSIFICATION_DIR

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "classification.yaml")
    parser.add_argument("--output-dir", type=Path, default=CLASSIFICATION_DIR / "screening")
    parser.add_argument("--folds", type=parse_folds, default=list(range(5)))
    parser.add_argument(
        "--limit-configs", type=int, default=None, help="Run only the first N (smoke tests)"
    )
    parser.add_argument("--fold-jobs", type=int, default=5, help="Folds fitted in parallel")
    parser.add_argument(
        "--model-jobs", type=int, default=None, help="Threads per model (default: cpus/fold-jobs)"
    )
    parser.add_argument("--overwrite", action="store_true")
    add_cohort_arguments(parser)
    add_tracking_arguments(parser)
    return parser.parse_args()


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _log_config(tracker: Tracker, scores: pd.DataFrame) -> None:
    first = scores.iloc[0]
    params = {
        column: first[column]
        for column in ("channel_set", "use_metadata", "selection", "k", "balancing", "model")
    }
    log_cv_child(tracker, str(first["config"]), scores, params=params)


def main() -> None:
    args = parse_args()
    settings = load_yaml(args.config)
    seed = int(settings["seed"])
    preprocessing = settings["preprocessing"]
    configs = screening_configs(settings["screening"], seed=seed, preprocessing=preprocessing)
    if args.limit_configs is not None:
        configs = configs[: args.limit_configs]
    model_jobs = args.model_jobs or max(1, (os.cpu_count() or 1) // args.fold_jobs)

    run_config = {
        "stage": "screening",
        "seed": seed,
        "preprocessing": preprocessing,
        "folds": args.folds,
        "configs": [config.name for config in configs],
    }
    output_dir = args.output_dir
    resumed = prepare_run_dir(output_dir, run_config, overwrite=args.overwrite)
    tracker = Tracker(
        enabled=not args.no_mlflow,
        experiment="classification-screening",
        output_dir=output_dir,
        run_name="screening",
        tracking_uri=args.mlflow_uri,
        params={
            **{key: value for key, value in run_config.items() if key != "configs"},
            "n_configs": len(configs),
            "screening": settings["screening"],
        },
        tags={"stage": "screening"},
    )
    with tracker:
        scores_path = output_dir / "fold_scores.csv"
        selected_path = output_dir / "selected_features.csv"
        done = set(pd.read_csv(scores_path)["config"]) if scores_path.exists() else set()
        if resumed:
            print(f"Resuming: {len(done)}/{len(configs)} configurations already evaluated")

        pending = [config for config in configs if config.name not in done]
        if pending:
            cohort = load_cohort(cohort_inputs(args))
            cache = FoldCache(cohort, preprocessing=preprocessing, folds=args.folds)
        for index, config in enumerate(pending, start=len(done) + 1):
            started = time.perf_counter()
            folds = cache.get(config.channel_set, config.use_metadata)
            results = evaluate_config(
                config, folds, fold_jobs=args.fold_jobs, model_jobs=model_jobs
            )
            if config.selection != "none":
                _append_csv(
                    pd.DataFrame(
                        [
                            {
                                "config": config.name,
                                "channel_set": config.channel_set,
                                "selection": config.selection,
                                "fold": result.fold,
                                "feature": feature,
                            }
                            for result in results
                            for feature in result.selected_features
                        ]
                    ),
                    selected_path,
                )
            # Scores are written last: a configuration counts as done only once complete.
            scores = fold_scores_frame(config, results)
            _append_csv(scores, scores_path)
            _log_config(tracker, scores)
            print(
                f"[{index}/{len(configs)}] {config.name}: "
                f"F1-macro {scores['f1_macro'].mean():.4f} ± {scores['f1_macro'].std():.4f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

        fold_scores = pd.read_csv(scores_path)
        for _, scores in fold_scores.groupby("config", sort=False):
            _log_config(tracker, scores)
        summary = summarize_fold_scores(fold_scores, SUMMARY_GROUPS)
        summary = summary.sort_values("f1_macro_mean", ascending=False)
        summary.to_csv(output_dir / "summary.csv", index=False)

        strategies = rank_strategies(summary)
        strategies.to_csv(output_dir / "strategy_ranking.csv", index=False)
        best = strategies.iloc[0]
        strategy = {
            "selection": best["selection"],
            "balancing": best["balancing"],
            "k": int(settings["screening"]["k"]),
            "mean_rank": float(best["mean_rank"]),
            "mean_f1_macro": float(best["mean_f1_macro"]),
            "complete": bool(best["complete"])
            and args.folds == list(range(5))
            and args.limit_configs is None
            and {config.name for config in configs} <= set(summary["config"]),
            "folds": args.folds,
            "rule": "lowest mean F1-macro rank across channel_set x model cells",
        }
        write_json(output_dir / "selected_strategy.json", strategy)

        frequency = jaccard = None
        if selected_path.exists():
            frequency, jaccard = feature_stability(pd.read_csv(selected_path))
            frequency.to_csv(output_dir / "feature_stability.csv", index=False)
            jaccard.to_csv(output_dir / "feature_stability_jaccard.csv", index=False)
            print("\nMean fold Jaccard of selected features:")
            print(jaccard.groupby(["channel_set", "selection"])["jaccard"].mean().round(3))

        figures_dir = output_dir / "figures"
        screening_figures(
            summary, strategies, figures_dir, frequency=frequency, jaccard=jaccard
        )
        tracker.set_tags(
            {
                "selected_selection": strategy["selection"],
                "selected_balancing": strategy["balancing"],
                "complete": strategy["complete"],
            }
        )
        tracker.log_metrics(numeric_metrics(strategy, prefix="selected_"))
        tracker.log_artifacts(
            [
                output_dir / name
                for name in (
                    "summary.csv",
                    "strategy_ranking.csv",
                    "selected_strategy.json",
                    "feature_stability.csv",
                    "feature_stability_jaccard.csv",
                )
            ]
            + [figures_dir]
        )

        print("\nStrategy ranking:")
        print(strategies.to_string(index=False))
        print(f"\nSelected: selection={best['selection']}, balancing={best['balancing']}")


if __name__ == "__main__":
    main()
