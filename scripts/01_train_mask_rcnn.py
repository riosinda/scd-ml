#!/usr/bin/env python3
"""Deprecated compatibility wrapper for :mod:`train_segmenter`."""

import warnings

from train_segmenter import main

if __name__ == "__main__":
    warnings.warn(
        "01_train_mask_rcnn.py is deprecated; use scripts/train_segmenter.py",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
