#!/usr/bin/env python3
"""Stage 4: refit the stage-2 winner on all development rows and score the frozen test.

The configuration comes exclusively from ``winner.json``; no hyperparameter can be
passed by hand. The test split is read once, never used for fitting, and existing
results are only replaced with ``--overwrite``. Figures go to ``<output-dir>/figures``
and the evaluation is logged as a new MLflow run unless ``--no-mlflow`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from scd_ml.classification.columns import CLASS_ORDER, encode_target
from scd_ml.classification.cv import PROBA_COLUMNS
from scd_ml.classification.data import coverage_by_class, load_cohort
from scd_ml.classification.metrics import (
    classification_metrics,
    confusion_frame,
    per_class_report,
)
from scd_ml.classification.pipeline import (
    ModelConfig,
    fit_full_pipeline,
    model_input_columns,
    selected_feature_names,
)
from scd_ml.classification.plots import holdout_figures
from scd_ml.classification.runs import (
    add_cohort_arguments,
    add_tracking_arguments,
    cohort_inputs,
    read_json,
)
from scd_ml.classification.tracking import Tracker, numeric_metrics
from scd_ml.data.schemas import assert_disjoint_groups
from scd_ml.paths import CLASSIFICATION_DIR, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, default=CLASSIFICATION_DIR / "tuning")
    parser.add_argument("--output-dir", type=Path, default=CLASSIFICATION_DIR / "test")
    parser.add_argument(
        "--model-path", type=Path, default=MODELS_DIR / "classification" / "winner.joblib"
    )
    parser.add_argument("--model-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    add_cohort_arguments(parser)
    add_tracking_arguments(parser)
    return parser.parse_args()


def load_winner(tuning_dir: Path) -> dict:
    path = tuning_dir / "winner.json"
    if not path.exists():
        raise FileNotFoundError(f"run scripts/tune_classifiers.py first: {path}")
    winner = read_json(path)
    if not winner.get("complete") or winner.get("folds") != list(range(5)):
        raise ValueError(
            f"{path} comes from an incomplete stage 2; the frozen test is only scored "
            "after screening and tuning finish on all five folds"
        )
    return winner


def guard_outputs(output_dir: Path, model_path: Path, *, overwrite: bool) -> None:
    existing = [
        path for path in (output_dir / "metrics.json", model_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"the frozen test was already scored ({existing}); use --overwrite to replace it"
        )


def main() -> None:
    args = parse_args()
    winner = load_winner(args.tuning_dir)
    guard_outputs(args.output_dir, args.model_path, overwrite=args.overwrite)
    config = ModelConfig.from_dict(winner["config"])

    cohort = load_cohort(cohort_inputs(args))
    eligible = cohort[cohort["eligible_for_classification"]]
    development = eligible[eligible["split"].eq("train")].reset_index(drop=True)
    test = eligible[eligible["split"].eq("test")].reset_index(drop=True)
    assert_disjoint_groups(pd.concat([development, test]))
    columns = model_input_columns(cohort.columns, config)

    print(f"Refitting {winner['study']} on {len(development):,} development images")
    pipeline = fit_full_pipeline(
        config,
        development[columns],
        encode_target(development["target"]),
        n_jobs=args.model_jobs,
    )
    y_test = encode_target(test["target"])
    proba = pipeline.predict_proba(test[columns])
    y_pred = proba.argmax(axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(y_test, proba)
    test_coverage = coverage_by_class(cohort[cohort["split"].eq("test")])
    payload = {
        "study": winner["study"],
        "config": config.to_dict(),
        "cv_metrics": winner["cv_metrics"],
        "test_metrics": metrics,
        "n_development": len(development),
        "n_test_scored": len(test),
        "n_test_excluded": int(test_coverage["excluded_images"].sum()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    per_class = per_class_report(y_test, y_pred)
    per_class.to_csv(args.output_dir / "per_class.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(args.output_dir / "confusion_matrix.csv")
    test_coverage.to_csv(args.output_dir / "coverage.csv", index=False)

    predictions = test[["image_id", "group_id", "target"]].copy()
    predictions["predicted"] = np.asarray(CLASS_ORDER)[y_pred]
    predictions[PROBA_COLUMNS] = proba
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)

    head_features = pipeline.named_steps["head"].get_feature_names_out()
    selected = selected_feature_names(pipeline, head_features)
    pd.DataFrame({"feature": selected}).to_csv(
        args.output_dir / "selected_features.csv", index=False
    )
    figures_dir = args.output_dir / "figures"
    holdout_figures(y_test, proba, per_class, figures_dir, selected_features=selected)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "config": config.to_dict(),
            "input_columns": columns,
            "class_order": list(CLASS_ORDER),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        args.model_path,
    )

    tracker = Tracker(
        enabled=not args.no_mlflow,
        experiment="classification-test",
        output_dir=args.output_dir,
        run_name=f"test_{winner['study']}",
        tracking_uri=args.mlflow_uri,
        params={
            **{key: value for key, value in config.to_dict().items() if key != "params"},
            "model_params": config.params,
            "study": winner["study"],
            "trial_number": winner["trial_number"],
            "n_development": len(development),
            "n_test_scored": len(test),
            "n_selected_features": len(selected),
        },
        tags={"stage": "test", "study": winner["study"]},
        resume=False,
    )
    with tracker:
        tracker.log_metrics(
            {
                **numeric_metrics(metrics, prefix="test_"),
                **numeric_metrics(winner["cv_metrics"], prefix="cv_"),
                "n_test_excluded": payload["n_test_excluded"],
            }
        )
        tracker.log_artifacts(
            [
                args.output_dir / name
                for name in (
                    "metrics.json",
                    "per_class.csv",
                    "confusion_matrix.csv",
                    "coverage.csv",
                    "selected_features.csv",
                    "predictions.parquet",
                )
            ]
            + [figures_dir, args.model_path]
        )

    print(json.dumps(metrics, indent=2))
    print(f"Excluded test images (no radiomics): {payload['n_test_excluded']:,}")
    print(f"Wrote {args.output_dir} and {args.model_path}")


if __name__ == "__main__":
    main()
