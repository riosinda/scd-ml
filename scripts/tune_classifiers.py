#!/usr/bin/env python3
"""Stage 2: tune every model per channel set with Optuna on the grouped folds.

Stage 2a tunes channel sets x models on radiomics only; stage 2b repeats the models
on the best 2a channel set with clinical metadata (age, sex, anatomical site). Both
use the (selection, balancing) pair chosen by stage 1. Studies live in a SQLite
database and resume automatically; only ``--overwrite`` discards them. Separate
processes may run ``--stage 2a --study <name>`` concurrently against the same
database. The report (``--stage all`` or ``report``) refits each study's best trial
on the folds to store out-of-fold probabilities, selected features, ablation tables,
corrected paired t-tests, figures (``<output-dir>/figures``) and ``winner.json``, and
mirrors every study to MLflow unless ``--no-mlflow`` is passed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import optuna
import pandas as pd

from scd_ml.classification.cv import FoldCache, evaluate_config, fold_scores_frame
from scd_ml.classification.data import load_cohort
from scd_ml.classification.metrics import PRIMARY_METRIC
from scd_ml.classification.pipeline import ModelConfig
from scd_ml.classification.plots import tuning_figures
from scd_ml.classification.reporting import (
    channel_ablation,
    friedman_over_models,
    metadata_ablation,
    pairwise_against,
)
from scd_ml.classification.runs import (
    add_cohort_arguments,
    add_tracking_arguments,
    cohort_inputs,
    load_yaml,
    parse_folds,
    prepare_run_dir,
    read_json,
    write_json,
)
from scd_ml.classification.screening import feature_stability
from scd_ml.classification.tracking import Tracker, log_cv_child, numeric_metrics
from scd_ml.classification.tuning import create_study, finished_trials, make_objective
from scd_ml.paths import CLASSIFICATION_DIR

ROOT = Path(__file__).resolve().parents[1]
METADATA_SUFFIX = "_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "classification.yaml")
    parser.add_argument("--screening-dir", type=Path, default=CLASSIFICATION_DIR / "screening")
    parser.add_argument("--output-dir", type=Path, default=CLASSIFICATION_DIR / "tuning")
    parser.add_argument("--stage", choices=["2a", "2b", "all", "report"], default="all")
    parser.add_argument(
        "--study", action="append", default=None, help="Only run these studies (repeatable)"
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override trials per study")
    parser.add_argument("--folds", type=parse_folds, default=list(range(5)))
    parser.add_argument(
        "--model-jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Threads per model (XGBoost, RandomForest and SVM calibration folds)",
    )
    parser.add_argument(
        "--allow-incomplete-screening",
        action="store_true",
        help="Accept a partial stage-1 result (smoke tests); the winner is marked incomplete",
    )
    parser.add_argument("--overwrite", action="store_true")
    add_cohort_arguments(parser)
    add_tracking_arguments(parser)
    return parser.parse_args()


class Stage2:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        settings = load_yaml(args.config)
        self.seed = int(settings["seed"])
        self.preprocessing = settings["preprocessing"]
        self.tuning = settings["tuning"]
        strategy_path = args.screening_dir / "selected_strategy.json"
        if not strategy_path.exists():
            raise FileNotFoundError(f"run scripts/screen_classifiers.py first: {strategy_path}")
        self.strategy = read_json(strategy_path)
        if not self.strategy["complete"] and not args.allow_incomplete_screening:
            raise ValueError(
                "stage-1 screening is incomplete; finish it or pass --allow-incomplete-screening"
            )
        self.output_dir = args.output_dir
        self.run_config = {
            "stage": "tuning",
            "seed": self.seed,
            "preprocessing": self.preprocessing,
            "strategy": {key: self.strategy[key] for key in ("selection", "balancing", "k")},
            "channel_sets": self.tuning["channel_sets"],
            "models": self.tuning["models"],
            "k_max": self.tuning["k_max"],
            "folds": args.folds,
        }
        prepare_run_dir(self.output_dir, self.run_config, overwrite=args.overwrite)
        self.storage = f"sqlite:///{(self.output_dir / 'optuna.db').resolve()}"
        self._cache: FoldCache | None = None

    @property
    def cache(self) -> FoldCache:
        if self._cache is None:
            cohort = load_cohort(cohort_inputs(self.args))
            self._cache = FoldCache(
                cohort, preprocessing=self.preprocessing, folds=self.args.folds
            )
        return self._cache

    def base_config(self, channel_set: str, model: str, use_metadata: bool) -> ModelConfig:
        selection = self.strategy["selection"]
        return ModelConfig(
            channel_set=channel_set,
            use_metadata=use_metadata,
            selection=selection,
            balancing=self.strategy["balancing"],
            model=model,
            k=int(self.strategy["k"]) if selection != "none" else None,
            seed=self.seed,
            preprocessing=dict(self.preprocessing),
        )

    @staticmethod
    def study_name(channel_set: str, model: str, use_metadata: bool) -> str:
        return f"{channel_set}_{model}{METADATA_SUFFIX if use_metadata else ''}"

    def run_study(self, base: ModelConfig) -> None:
        name = self.study_name(base.channel_set, base.model, base.use_metadata)
        if self.args.study and name not in self.args.study:
            return
        pruner = self.tuning["pruner"]
        study = create_study(
            name,
            self.storage,
            seed=self.seed,
            n_startup_trials=int(pruner["n_startup_trials"]),
            n_warmup_steps=int(pruner["n_warmup_steps"]),
        )
        target = self.args.n_trials or int(self.tuning["n_trials"][base.model])
        remaining = target - finished_trials(study)
        if remaining <= 0:
            print(f"{name}: {finished_trials(study)} trials already finished")
            return
        print(f"{name}: running {remaining} trials", flush=True)
        folds = self.cache.get(base.channel_set, base.use_metadata)
        objective = make_objective(
            base, folds, k_max=int(self.tuning["k_max"]), model_jobs=self.args.model_jobs
        )
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    def best_values(self) -> pd.DataFrame:
        rows = []
        for summary in optuna.get_all_study_summaries(self.storage, include_best_trial=True):
            if summary.best_trial is None:
                continue
            config = summary.best_trial.user_attrs["config"]
            rows.append(
                {
                    "study": summary.study_name,
                    "channel_set": config["channel_set"],
                    "use_metadata": config["use_metadata"],
                    "model": config["model"],
                    "best_value": summary.best_trial.value,
                    "best_trial": summary.best_trial.number,
                    "n_trials": summary.n_trials,
                }
            )
        columns = ["study", "channel_set", "use_metadata", "model", "best_value"]
        return pd.DataFrame(rows, columns=[*columns, "best_trial", "n_trials"])

    def stage_2a(self) -> None:
        for channel_set in self.tuning["channel_sets"]:
            for model in self.tuning["models"]:
                self.run_study(self.base_config(channel_set, model, use_metadata=False))

    def stage_2b_channel(self) -> str | None:
        values = self.best_values()
        radiomics = values[~values["use_metadata"].astype(bool)]
        expected = {
            self.study_name(channel_set, model, False)
            for channel_set in self.tuning["channel_sets"]
            for model in self.tuning["models"]
        }
        missing = sorted(expected - set(radiomics["study"]))
        if missing:
            print(f"Stage 2b waits for completed stage-2a studies: {missing}")
            return None
        by_channel = radiomics.groupby("channel_set")["best_value"].mean()
        return str(by_channel.idxmax())

    def stage_2b(self) -> None:
        channel_set = self.stage_2b_channel()
        if channel_set is None:
            return
        print(f"Stage 2b uses the best stage-2a channel set: {channel_set}")
        for model in self.tuning["models"]:
            self.run_study(self.base_config(channel_set, model, use_metadata=True))

    def refit_best(self, study_name: str) -> pd.DataFrame:
        """Evaluate the best trial on every fold and store its OOF probabilities."""
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        best = study.best_trial
        params_path = self.output_dir / "best_params" / f"{study_name}.json"
        oof_path = self.output_dir / "oof" / f"{study_name}.parquet"
        scores_path = self.output_dir / "cv" / f"{study_name}.csv"
        selected_path = self.output_dir / "selected_features" / f"{study_name}.csv"
        outputs = (params_path, oof_path, scores_path, selected_path)
        if all(path.exists() for path in outputs):
            if read_json(params_path)["trial_number"] == best.number:
                return pd.read_csv(scores_path)

        config = ModelConfig.from_dict(best.user_attrs["config"])
        folds = self.cache.get(config.channel_set, config.use_metadata)
        results = evaluate_config(config, folds, model_jobs=self.args.model_jobs)
        scores = fold_scores_frame(config, results)
        scores.insert(0, "study", study_name)
        oof = pd.concat([result.oof for result in results], ignore_index=True)
        oof.insert(0, "study", study_name)

        selected = pd.DataFrame(
            [
                {
                    "config": study_name,
                    "channel_set": config.channel_set,
                    "selection": config.selection,
                    "fold": result.fold,
                    "feature": feature,
                }
                for result in results
                for feature in result.selected_features
            ]
        )

        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scores_path, index=False)
        oof.to_parquet(oof_path, index=False)
        selected.to_csv(selected_path, index=False)
        write_json(
            params_path,
            {
                "study": study_name,
                "trial_number": best.number,
                "optuna_value": best.value,
                "config": config.to_dict(),
            },
        )
        return scores

    def report(self) -> None:
        values = self.best_values()
        if values.empty:
            print("No completed trials to report")
            return
        fold_scores = pd.concat(
            [self.refit_best(study) for study in values["study"]], ignore_index=True
        )
        fold_scores.to_csv(self.output_dir / "fold_scores.csv", index=False)

        metric_columns = [
            column
            for column in fold_scores.columns
            if column in {"f1_macro", "balanced_accuracy", "roc_auc_ovr_macro"}
            or column.startswith("recall__")
        ]
        grouped = fold_scores.groupby("study")
        summary = grouped[metric_columns].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = values.set_index("study").join(summary).reset_index()
        summary["k"] = grouped["k"].first().reindex(summary["study"]).to_numpy()
        summary = summary.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)
        summary.to_csv(self.output_dir / "studies_summary.csv", index=False)

        channels = channel_ablation(fold_scores)
        channels.to_csv(self.output_dir / "channel_ablation.csv", index=False)
        metadata_channels = sorted(
            fold_scores.loc[fold_scores["use_metadata"].astype(bool), "channel_set"].unique()
        )
        metadata_tables = [metadata_ablation(fold_scores, channel) for channel in metadata_channels]
        metadata = pd.concat(metadata_tables, ignore_index=True) if metadata_tables else None
        if metadata is not None:
            metadata.to_csv(self.output_dir / "metadata_ablation.csv", index=False)

        winner = str(summary.iloc[0]["study"])
        if len(summary) > 1:
            pairwise_against(fold_scores, winner).to_csv(
                self.output_dir / "pairwise_tests.csv", index=False
            )
        winner_params = read_json(self.output_dir / "best_params" / f"{winner}.json")
        expected_studies = len(self.tuning["channel_sets"]) * len(self.tuning["models"]) + len(
            self.tuning["models"]
        )
        complete = (
            bool(self.strategy["complete"])
            and self.args.folds == list(range(5))
            and len(values) == expected_studies
        )
        winner_row = summary.iloc[0]
        cv_metrics = {
            column: float(winner_row[column])
            for column in summary.columns
            if column.endswith(("_mean", "_std"))
        }
        write_json(
            self.output_dir / "winner.json",
            {
                "study": winner,
                "trial_number": winner_params["trial_number"],
                "config": winner_params["config"],
                "cv_metrics": cv_metrics,
                "folds": self.args.folds,
                "complete": complete,
                "rule": "highest mean CV F1-macro across all stage-2 studies",
            },
        )
        winner_channel = str(winner_row["channel_set"])
        write_json(
            self.output_dir / "tuning_report.json",
            {
                "winner": winner,
                "complete": complete,
                "n_studies": len(values),
                "expected_studies": expected_studies,
                "metadata_channel_sets": metadata_channels,
                "friedman_models_on_winner_channel": friedman_over_models(
                    fold_scores, winner_channel
                ),
            },
        )
        print(summary[["study", f"{PRIMARY_METRIC}_mean", f"{PRIMARY_METRIC}_std"]].to_string(
            index=False
        ))
        print(f"\nWinner: {winner} (complete={complete})")

        figures_dir = self.output_dir / "figures"
        winner_selection = None
        if winner_params["config"]["selection"] != "none":
            selected = pd.read_csv(self.output_dir / "selected_features" / f"{winner}.csv")
            winner_selection, _ = feature_stability(selected)
        tuning_figures(
            summary,
            figures_dir,
            channel_ablation=channels,
            metadata_ablation=metadata,
            history=self.history(),
            winner_selection=winner_selection,
            winner=winner,
        )
        self.track(values, fold_scores, winner, complete, cv_metrics, figures_dir)

    def history(self) -> pd.DataFrame:
        """Completed trials of every study with their objective value."""
        rows = []
        for summary in optuna.get_all_study_summaries(self.storage, include_best_trial=False):
            study = optuna.load_study(study_name=summary.study_name, storage=self.storage)
            states = (optuna.trial.TrialState.COMPLETE,)
            for trial in study.get_trials(deepcopy=False, states=states):
                rows.append(
                    {
                        "study": summary.study_name,
                        "model": trial.user_attrs["config"]["model"],
                        "number": trial.number,
                        "value": trial.value,
                    }
                )
        return pd.DataFrame(rows, columns=["study", "model", "number", "value"])

    def track(
        self,
        values: pd.DataFrame,
        fold_scores: pd.DataFrame,
        winner: str,
        complete: bool,
        cv_metrics: dict[str, float],
        figures_dir: Path,
    ) -> None:
        """Mirror every study's best trial and the stage-2 report to MLflow."""
        tracker = Tracker(
            enabled=not self.args.no_mlflow,
            experiment="classification-tuning",
            output_dir=self.output_dir,
            run_name="tuning",
            tracking_uri=self.args.mlflow_uri,
            params=self.run_config,
            tags={"stage": "tuning"},
        )
        with tracker:
            for row in values.itertuples(index=False):
                params_path = self.output_dir / "best_params" / f"{row.study}.json"
                config = read_json(params_path)["config"]
                log_cv_child(
                    tracker,
                    row.study,
                    fold_scores[fold_scores["study"].eq(row.study)],
                    params={
                        **{key: value for key, value in config.items() if key != "params"},
                        "model_params": config["params"],
                        "best_trial": row.best_trial,
                    },
                    tags={"model": row.model, "channel_set": row.channel_set},
                    artifacts=[
                        params_path,
                        self.output_dir / "cv" / f"{row.study}.csv",
                        self.output_dir / "selected_features" / f"{row.study}.csv",
                    ],
                    # A better trial found by later optimisation replaces the child run.
                    version=f"trial-{row.best_trial}",
                )
            tracker.set_tags({"winner": winner, "complete": complete})
            tracker.log_metrics(numeric_metrics(cv_metrics, prefix="winner_"))
            tracker.log_artifacts(
                [
                    self.output_dir / name
                    for name in (
                        "winner.json",
                        "tuning_report.json",
                        "studies_summary.csv",
                        "channel_ablation.csv",
                        "metadata_ablation.csv",
                        "pairwise_tests.csv",
                    )
                ]
                + [figures_dir]
            )


def main() -> None:
    args = parse_args()
    stage2 = Stage2(args)
    if args.stage in {"2a", "all"}:
        stage2.stage_2a()
    if args.stage in {"2b", "all"}:
        stage2.stage_2b()
    # Parallel processes may run single stages; only one final call should report.
    if args.stage in {"all", "report"}:
        stage2.report()


if __name__ == "__main__":
    main()
