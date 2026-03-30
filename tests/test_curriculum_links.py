from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_curriculum_links import _iter_links


class TestCurriculumLinks(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(ROOT / "scripts"))

    def test_link_checker_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_curriculum_links.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)

    def test_iter_links_parses_empty_alt_image_links(self) -> None:
        links = _iter_links("![](figures/example.svg)\n[doc](README.md)")

        self.assertEqual(["figures/example.svg", "README.md"], links)


if __name__ == "__main__":
    unittest.main()
