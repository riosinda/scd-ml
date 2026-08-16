from __future__ import annotations

import ast
import unittest
from pathlib import Path


class Python37BoundaryTests(unittest.TestCase):
    def test_standalone_extractor_uses_python37_compatible_interface(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for relative in ("scripts/extract_radiomics.py",):
            source = (root / relative).read_text(encoding="utf-8")
            ast.parse(source, filename=relative, feature_version=(3, 7))
            self.assertNotIn("from scd_ml", source)
            self.assertNotIn("from __future__ import annotations", source)


if __name__ == "__main__":
    unittest.main()
