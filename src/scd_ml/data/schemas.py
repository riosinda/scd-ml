"""Shared validation and file I/O for pipeline manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_unique(frame: pd.DataFrame, column: str, *, name: str) -> None:
    duplicated = frame.loc[frame[column].duplicated(keep=False), column]
    if not duplicated.empty:
        examples = duplicated.astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"{name}.{column} must be unique; examples: {examples}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format {suffix!r}; use CSV or Parquet")


def write_manifest(frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def assert_disjoint_groups(
    frame: pd.DataFrame,
    *,
    group_column: str = "group_id",
    split_column: str = "split",
) -> None:
    memberships = frame.groupby(group_column, dropna=False)[split_column].nunique()
    leaked = memberships[memberships > 1]
    if not leaked.empty:
        examples = leaked.index.astype(str).tolist()[:5]
        raise ValueError(f"Groups occur in multiple splits: {examples}")
