"""Delete ISIC archive images that did not pass the EDA selection filters.

The ISIC archive ships with ~550k images but only ~76k pass the filters defined
in notebooks/00_eda_isci_archive.ipynb (4 target classes + melanocytic not null).
The notebook saves the kept image IDs to results/processed/isic_selected.parquet;
this script uses that parquet as the source of truth to prune unused images
from disk and reclaim storage.

Usage (from repo root):
    # Dry run — reports what would be deleted, touches nothing
    python scripts/prune_isic_images.py

    # Actually delete
    python scripts/prune_isic_images.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from tqdm import tqdm

from src.paths import ISIC_IMAGES_DIR, PROCESSED_DIR

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--parquet',
        type=Path,
        default=PROCESSED_DIR / 'isic_selected.parquet',
        help='Parquet file with the kept isic_id list (default: %(default)s)',
    )
    p.add_argument(
        '--images-dir',
        type=Path,
        default=ISIC_IMAGES_DIR,
        help='Directory containing ISIC images (default: %(default)s)',
    )
    p.add_argument(
        '--apply',
        action='store_true',
        help='Actually delete files. Without this flag, runs as a dry-run.',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.parquet.exists():
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        print("Run notebooks/00_eda_isci_archive.ipynb first to generate it.", file=sys.stderr)
        return 1

    if not args.images_dir.exists():
        print(f"ERROR: images directory not found at {args.images_dir}", file=sys.stderr)
        return 1

    selected_ids = set(pl.read_parquet(args.parquet)['isic_id'].to_list())
    on_disk_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(IMAGE_EXTS)]
    on_disk_ids = {os.path.splitext(f)[0]: f for f in on_disk_files}

    kept = selected_ids & on_disk_ids.keys()
    not_found = selected_ids - on_disk_ids.keys()
    unused = on_disk_ids.keys() - selected_ids

    unused_bytes = sum(
        (args.images_dir / on_disk_ids[img_id]).stat().st_size
        for img_id in unused
    )
    unused_gb = unused_bytes / 1024**3

    print(f"Parquet:         {args.parquet}")
    print(f"Images dir:      {args.images_dir}")
    print()
    print(f"Selected (parquet):  {len(selected_ids):,}")
    print(f"On disk:             {len(on_disk_ids):,}")
    print(f"Will be kept:        {len(kept):,}")
    print(f"Missing on disk:     {len(not_found):,}  (in parquet but no file)")
    print(f"Unused on disk:      {len(unused):,}  ({unused_gb:.2f} GB)")
    print()

    if not unused:
        print("Nothing to delete.")
        return 0

    if not args.apply:
        print("Dry-run only — no files deleted. Re-run with --apply to delete.")
        return 0

    print(f"Deleting {len(unused):,} files…")
    deleted = 0
    freed_bytes = 0
    failed: list[tuple[str, str]] = []
    for img_id in tqdm(unused, desc='Deleting'):
        path = args.images_dir / on_disk_ids[img_id]
        try:
            size = path.stat().st_size
            path.unlink()
            deleted += 1
            freed_bytes += size
        except OSError as e:
            failed.append((img_id, str(e)))

    print()
    print(f"Deleted {deleted:,} files ({freed_bytes / 1024**3:.2f} GB freed)")
    if failed:
        print(f"Failed to delete {len(failed)} files. First few: {failed[:5]}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
