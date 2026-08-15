#!/usr/bin/env python3
"""Deprecated compatibility wrapper for :mod:`segment_isic`."""

import warnings

from segment_isic import main

if __name__ == "__main__":
    warnings.warn(
        "02_apply_segmentation.py is deprecated; use scripts/segment_isic.py",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
