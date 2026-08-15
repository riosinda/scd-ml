#!/usr/bin/env python3
"""Validate the radiomics CSV outputs before classification work begins."""

from __future__ import annotations

import argparse
from pathlib import Path

from scd_ml.features.radiomics_contract import validate_radiomics_contract
from scd_ml.paths import FEATURES_DIR, SEGMENTATION_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masks-manifest", type=Path, default=SEGMENTATION_DIR / "isic_masks_manifest.csv"
    )
    parser.add_argument(
        "--features", type=Path, default=FEATURES_DIR / "radiomics_features.csv"
    )
    parser.add_argument("--status", type=Path, default=FEATURES_DIR / "radiomics_status.csv")
    args = parser.parse_args()
    summary = validate_radiomics_contract(args.masks_manifest, args.features, args.status)
    for key, value in summary.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
