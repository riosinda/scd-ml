"""Run bookkeeping shared by the classification entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from .data import CohortInputs


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_run_dir(output_dir: Path, run_config: dict[str, Any], *, overwrite: bool) -> bool:
    """Create or resume ``output_dir``; return whether an earlier run is resumed.

    Only ``overwrite`` deletes existing progress. Resuming with a different run
    configuration is refused so results from different protocols never mix.
    """
    output_dir = Path(output_dir)
    marker = output_dir / "run_config.json"
    normalized = json.loads(json.dumps(run_config))
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if read_json(marker) != normalized:
            raise ValueError(
                f"{marker} was written with a different configuration; "
                "use --overwrite to discard the previous run"
            )
        return True
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty and has no run_config.json; use --overwrite"
        )
    write_json(marker, normalized)
    return False


def parse_folds(value: str) -> list[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or not set(folds) <= set(range(5)):
        raise argparse.ArgumentTypeError("folds must be a comma-separated subset of 0-4")
    return folds


def add_cohort_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = CohortInputs()
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument("--masks-manifest", type=Path, default=defaults.masks_manifest)
    parser.add_argument("--features", type=Path, default=defaults.features)
    parser.add_argument("--status", type=Path, default=defaults.status)
    parser.add_argument("--metadata", type=Path, default=defaults.metadata)


def cohort_inputs(args: argparse.Namespace) -> CohortInputs:
    return CohortInputs(
        manifest=args.manifest,
        masks_manifest=args.masks_manifest,
        features=args.features,
        status=args.status,
        metadata=args.metadata,
    )


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or results/mlflow/mlflow.db)",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Do not mirror results to MLflow"
    )
