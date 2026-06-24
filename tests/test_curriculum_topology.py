from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CURRICULUM_LADDER = [
    "00_foundations",
    "01_ml",
    "02_deep_learning",
    "03_nlp_bridge",
    "04_nlp",
    "05_advanced_nlp_llm",
    "06_training_systems",
    "07_frontier_labs",
    "08_multimodal_bridge",
    "09_multimodal",
    "10_vla",
]


def assert_tokens_appear_in_order(testcase: unittest.TestCase, text: str, tokens: list[str]) -> None:
    cursor = 0
    for token in tokens:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
        match = re.search(pattern, text[cursor:])
        testcase.assertIsNotNone(match, f"missing ordered curriculum token: {token}")
        cursor += match.end()


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
            ("10_vla", "10_vla/README.md"),
        ]

        positions: list[int] = []
        for label, href in ladder:
            pattern = rf"\[[^\]]*{re.escape(label)}[^\]]*\]\({re.escape(href)}\)"
            match = re.search(pattern, text)
            self.assertIsNotNone(match, f"README.md missing link for {label}")
            positions.append(match.start())

        self.assertEqual(positions, sorted(positions), "curriculum ladder order changed")

    def test_program_map_mentions_full_future_ladder_in_order_and_language_policy(self) -> None:
        text = (ROOT / "docs" / "00_program_map.md").read_text(encoding="utf-8")
        assert_tokens_appear_in_order(self, text, CANONICAL_CURRICULUM_LADDER)
        self.assertRegex(text, r"(?s)(한글|한국어).*(우선|중심)")

    def test_new_entry_dirs_have_readmes(self) -> None:
        for rel in ["00_foundations", "02_deep_learning", "03_nlp_bridge", "08_multimodal_bridge"]:
            path = ROOT / rel / "README.md"
            self.assertTrue(path.exists(), f"missing {rel}/README.md")

    def test_superpowers_artifacts_are_ignored(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".superpowers/", text)


if __name__ == "__main__":
    unittest.main()
