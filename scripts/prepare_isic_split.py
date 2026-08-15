#!/usr/bin/env python3
"""Create the locked ISIC classification split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from scd_ml.data.isic_splits import build_isic_manifest
from scd_ml.data.schemas import read_table, write_manifest
from scd_ml.paths import ISIC_DIR, SPLITS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ISIC_DIR / "metadata.csv")
    parser.add_argument("--output", type=Path, default=SPLITS_DIR / "isic_classification.csv")
    parser.add_argument("--image-id-column", default="isic_id")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = read_table(args.metadata)
    manifest = build_isic_manifest(
        metadata,
        image_id_column=args.image_id_column,
        target_column=args.target_column,
        seed=args.seed,
    )
    output = write_manifest(manifest, args.output, overwrite=args.overwrite)
    print(manifest["split"].value_counts().to_string())
    print(f"Wrote {len(manifest):,} rows to {output}")


if __name__ == "__main__":
    main()
