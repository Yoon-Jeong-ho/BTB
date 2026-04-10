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
            ("02_deep_learning", "02_deep_learning/README.md"),
            ("03_nlp_bridge", "03_nlp_bridge/README.md"),
            ("04_nlp", "04_nlp/README.md"),
            ("05_advanced_nlp_llm", "05_advanced_nlp_llm/README.md"),
            ("06_training_systems", "06_training_systems/README.md"),
            ("07_frontier_labs", "07_frontier_labs/README.md"),
            ("08_multimodal_bridge", "08_multimodal_bridge/README.md"),
            ("09_multimodal", "09_multimodal/README.md"),
        ]

        positions: list[int] = []
        for label, href in ladder:
            pattern = rf"\[[^\]]*{re.escape(label)}[^\]]*\]\({re.escape(href)}\)"
            match = re.search(pattern, text)
            self.assertIsNotNone(match, f"README.md missing link for {label}")
            positions.append(match.start())

        self.assertEqual(positions, sorted(positions), "curriculum ladder order changed")

    def test_program_map_mentions_new_bridge_positions_and_language_policy(self) -> None:
        text = (ROOT / "docs" / "00_program_map.md").read_text(encoding="utf-8")
        for rel in ["00_foundations", "03_nlp_bridge", "08_multimodal_bridge"]:
            self.assertIn(rel, text)
        self.assertLess(text.index("03_nlp_bridge"), text.index("08_multimodal_bridge"))
        self.assertRegex(text, r"(한글|한국어).*(우선|중심)")

    def test_new_entry_dirs_have_readmes(self) -> None:
        for rel in ["00_foundations", "02_deep_learning", "03_nlp_bridge", "08_multimodal_bridge"]:
            path = ROOT / rel / "README.md"
            self.assertTrue(path.exists(), f"missing {rel}/README.md")

    def test_superpowers_artifacts_are_ignored(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".superpowers/", text)


if __name__ == "__main__":
    unittest.main()
