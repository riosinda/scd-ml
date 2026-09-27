"""Optional MLflow mirror of the classification results.

The files under ``results/classification`` remain the source of truth: ``winner.json``
alone decides what the frozen test scores. MLflow indexes the same numbers so runs
can be browsed and compared with ``mlflow ui``.

Each stage owns one parent run whose id is stored in ``<output_dir>/mlflow_run.json``.
A resumed stage reopens that parent and only logs children it does not have yet, so
interrupted runs never duplicate entries; ``--overwrite`` empties the output
directory and therefore starts a new parent run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from scd_ml.paths import MLFLOW_DIR

from .cv import summarize_fold_scores
from .metrics import PRIMARY_METRIC
from .runs import read_json, write_json

RUN_FILE = "mlflow_run.json"
KEY_TAG = "scd.key"
VERSION_TAG = "scd.version"
PARENT_TAG = "mlflow.parentRunId"
MAX_PARAM_LENGTH = 6000
BATCH_SIZE = 500


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """``--mlflow-uri``, then ``MLFLOW_TRACKING_URI``, then a local SQLite store."""
    if explicit:
        return explicit
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').resolve()}"


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Nested mapping to dotted MLflow params with string values."""
    flat: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:MAX_PARAM_LENGTH]
    return flat


def numeric_metrics(values: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Keep finite-able numeric entries (bools excluded) as MLflow metrics."""
    metrics = {}
    for key, value in values.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            metrics[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


class Tracker:
    """Parent run of one stage plus idempotent child runs; a no-op when disabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        experiment: str,
        output_dir: Path,
        run_name: str,
        tracking_uri: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
        resume: bool = True,
    ) -> None:
        self.enabled = enabled
        self.run_id: str | None = None
        self._children: dict[str, tuple[str, str]] = {}
        if not enabled:
            return
        from mlflow.tracking import MlflowClient

        uri = resolve_tracking_uri(tracking_uri)
        if uri.startswith("sqlite:///"):
            Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.uri = uri
        self.client = MlflowClient(tracking_uri=uri)
        self.experiment_id = self._experiment_id(experiment)

        run_file = Path(output_dir) / RUN_FILE
        run_id = read_json(run_file)["run_id"] if resume and run_file.exists() else None
        if run_id is not None and not self._is_active(run_id):
            run_id = None
        if run_id is None:
            run = self.client.create_run(
                self.experiment_id, run_name=run_name, tags=dict(tags or {})
            )
            run_id = run.info.run_id
            self._log(run_id, params=flatten(params or {}))
            write_json(
                run_file,
                {"run_id": run_id, "experiment": experiment, "tracking_uri": uri},
            )
        else:
            self.client.update_run(run_id, status="RUNNING")
            self._children = self._existing_children(run_id)
        self.run_id = run_id
        print(f"MLflow: run {run_id} in experiment '{experiment}' ({uri})")

    # ----------------------------------------------------------------- internals

    def _experiment_id(self, name: str) -> str:
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            if experiment.lifecycle_stage == "deleted":
                self.client.restore_experiment(experiment.experiment_id)
            return experiment.experiment_id
        location = None
        if self.uri.startswith("sqlite:///"):
            database = Path(self.uri.removeprefix("sqlite:///"))
            location = (database.parent / "artifacts").resolve().as_uri()
        return self.client.create_experiment(name, artifact_location=location)

    def _is_active(self, run_id: str) -> bool:
        try:
            return self.client.get_run(run_id).info.lifecycle_stage == "active"
        except Exception:  # noqa: BLE001 - a missing run simply starts a new parent
            return False

    def _existing_children(self, parent_id: str) -> dict[str, tuple[str, str]]:
        children: dict[str, tuple[str, str]] = {}
        token = None
        while True:
            page = self.client.search_runs(
                [self.experiment_id],
                filter_string=f"tags.`{PARENT_TAG}` = '{parent_id}'",
                max_results=1000,
                page_token=token,
            )
            for run in page:
                key = run.data.tags.get(KEY_TAG)
                if key is not None:
                    children[key] = (run.info.run_id, run.data.tags.get(VERSION_TAG, ""))
            token = page.token
            if not token:
                return children

    def _log(
        self,
        run_id: str,
        *,
        params: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
        steps: Iterable[tuple[str, float, int]] = (),
        tags: Mapping[str, str] | None = None,
    ) -> None:
        from mlflow.entities import Metric, Param, RunTag

        timestamp = int(time.time() * 1000)
        entries: list[tuple[str, Any]] = []
        entries += [("param", Param(k, v)) for k, v in (params or {}).items()]
        entries += [("tag", RunTag(k, str(v))) for k, v in (tags or {}).items()]
        entries += [
            ("metric", Metric(k, float(v), timestamp, 0)) for k, v in (metrics or {}).items()
        ]
        entries += [("metric", Metric(k, float(v), timestamp, int(step))) for k, v, step in steps]
        for start in range(0, len(entries), BATCH_SIZE):
            chunk = entries[start : start + BATCH_SIZE]
            self.client.log_batch(
                run_id,
                metrics=[value for kind, value in chunk if kind == "metric"],
                params=[value for kind, value in chunk if kind == "param"],
                tags=[value for kind, value in chunk if kind == "tag"],
            )

    def _artifacts(self, run_id: str, paths: Iterable[Path]) -> None:
        for path in paths:
            path = Path(path)
            if path.is_dir():
                self.client.log_artifacts(run_id, str(path), artifact_path=path.name)
            elif path.exists():
                self.client.log_artifact(run_id, str(path))

    # -------------------------------------------------------------------- public

    def has_child(self, key: str, version: str = "") -> bool:
        return key in self._children and self._children[key][1] == version

    def log_child(
        self,
        key: str,
        *,
        run_name: str,
        params: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        steps: Iterable[tuple[str, float, int]] = (),
        tags: Mapping[str, str] | None = None,
        artifacts: Iterable[Path] = (),
        version: str = "",
    ) -> None:
        """Log one nested run once per ``key``; a new ``version`` replaces the old run."""
        if not self.enabled or self.has_child(key, version):
            return
        if key in self._children:
            self.client.delete_run(self._children[key][0])
        run = self.client.create_run(
            self.experiment_id,
            run_name=run_name,
            tags={PARENT_TAG: self.run_id, KEY_TAG: key, VERSION_TAG: version, **(tags or {})},
        )
        child_id = run.info.run_id
        self._log(child_id, params=flatten(params or {}), metrics=metrics, steps=steps)
        self._artifacts(child_id, artifacts)
        self.client.set_terminated(child_id)
        self._children[key] = (child_id, version)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        if self.enabled:
            self._log(self.run_id, metrics=metrics)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.enabled:
            self._log(self.run_id, tags={key: str(value) for key, value in tags.items()})

    def log_artifacts(self, paths: Iterable[Path]) -> None:
        if self.enabled:
            self._artifacts(self.run_id, paths)

    def finish(self, status: str = "FINISHED") -> None:
        if self.enabled and self.run_id is not None:
            self.client.set_terminated(self.run_id, status=status)

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finish("FAILED" if exc_type is not None else "FINISHED")


def log_cv_child(
    tracker: Tracker,
    key: str,
    fold_scores: pd.DataFrame,
    *,
    params: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
    artifacts: Iterable[Path] = (),
    version: str = "",
) -> None:
    """Child run with mean/sd of every CV metric and per-fold F1-macro as steps."""
    if not tracker.enabled or tracker.has_child(key, version):
        return
    summary = summarize_fold_scores(fold_scores.assign(_key=key), ["_key"]).iloc[0]
    metrics = numeric_metrics(summary.drop(["_key"]))
    steps = [
        (f"fold_{PRIMARY_METRIC}", float(row[PRIMARY_METRIC]), int(row["fold"]))
        for _, row in fold_scores.iterrows()
    ]
    tracker.log_child(
        key,
        run_name=key,
        params=params,
        metrics=metrics,
        steps=steps,
        tags=tags,
        artifacts=artifacts,
        version=version,
    )
