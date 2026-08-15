#!/usr/bin/env python3
"""Create the lesion-grouped HAM10000 segmentation split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from scd_ml.data.ham10000_splits import build_ham10000_manifest
from scd_ml.data.schemas import read_table, write_manifest
from scd_ml.paths import HAM10000_DIR, SPLITS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=HAM10000_DIR / "metadata.csv")
    parser.add_argument(
        "--output", type=Path, default=SPLITS_DIR / "ham10000_segmentation.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_ham10000_manifest(read_table(args.metadata), seed=args.seed)
    output = write_manifest(manifest, args.output, overwrite=args.overwrite)
    print(manifest["split"].value_counts().to_string())
    print(f"Wrote {len(manifest):,} rows to {output}")


if __name__ == "__main__":
    main()
