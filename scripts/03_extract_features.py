#!/usr/bin/env python3
"""Deprecated Python 3.7 wrapper for ``extract_radiomics.py``."""

import warnings

from extract_radiomics import main

if __name__ == "__main__":
    warnings.warn(
        "03_extract_features.py is deprecated; use scripts/extract_radiomics.py",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
