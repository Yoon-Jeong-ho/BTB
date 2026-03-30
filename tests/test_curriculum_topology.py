from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumTopology(unittest.TestCase):
    def test_root_readme_mentions_new_ladder_in_order(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        ladder = [
            ("00_foundations", "00_foundations/README.md"),
            ("01_ml", "01_ml/README.md"),
            ("02_nlp_bridge", "02_nlp_bridge/README.md"),
            ("03_nlp", "03_nlp/README.md"),
            ("04_multimodal_bridge", "04_multimodal_bridge/README.md"),
            ("05_multimodal", "05_multimodal/README.md"),
        ]

        positions: list[int] = []
        for label, href in ladder:
            pattern = rf"\[[^\]]*{re.escape(label)}[^\]]*\]\({re.escape(href)}\)"
            match = re.search(pattern, text)
            self.assertIsNotNone(match, f"README.md missing link for {label}")
            positions.append(match.start())

        self.assertEqual(positions, sorted(positions), "curriculum ladder order changed")

    def test_program_map_mentions_foundations_bridges_and_language_policy(self) -> None:
        text = (ROOT / "docs" / "00_program_map.md").read_text(encoding="utf-8")
        for rel in ["00_foundations", "02_nlp_bridge", "04_multimodal_bridge"]:
            self.assertIn(rel, text)
        self.assertRegex(text, r"(한글|한국어).*(우선|중심)")

    def test_new_entry_dirs_have_korean_readmes(self) -> None:
        for rel in ["00_foundations", "02_nlp_bridge", "04_multimodal_bridge"]:
            path = ROOT / rel / "README.md"
            self.assertTrue(path.exists(), f"missing {rel}/README.md")
            self.assertRegex(path.read_text(encoding="utf-8"), r"[가-힣]")

    def test_superpowers_artifacts_are_ignored(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".superpowers/", text)


if __name__ == "__main__":
    unittest.main()
