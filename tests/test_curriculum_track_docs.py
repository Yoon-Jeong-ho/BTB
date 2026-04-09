from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumTrackDocs(unittest.TestCase):
    def test_root_readme_removes_placeholder_language(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")

        for stale in ["재배치 대상", "자리를 미리 고정", "단계적으로 재배치 중"]:
            self.assertNotIn(stale, text)

        self.assertIn("foundations/bridge/applied", text)
        self.assertIn("03_nlp", text)
        self.assertIn("05_multimodal", text)

    def test_reindexed_track_headings_match_directory_numbers(self) -> None:
        self.assertEqual(
            (ROOT / "03_nlp" / "README.md").read_text(encoding="utf-8").splitlines()[0],
            "# 03 NLP",
        )
        self.assertEqual(
            (ROOT / "05_multimodal" / "README.md").read_text(encoding="utf-8").splitlines()[0],
            "# 05 Multimodal",
        )

    def test_track_readmes_describe_real_populated_units(self) -> None:
        nlp_text = (ROOT / "03_nlp" / "README.md").read_text(encoding="utf-8")
        multimodal_text = (ROOT / "05_multimodal" / "README.md").read_text(encoding="utf-8")

        for token in [
            "01_text_classification",
            "02_named_entity_recognition",
            "03_machine_reading_comprehension",
            "세 unit로 채워져 있으며",
        ]:
            self.assertIn(token, nlp_text)

        for token in [
            "01_image_text_retrieval",
            "02_image_captioning",
            "03_visual_question_answering",
            "세 unit로 채워져 있으며",
        ]:
            self.assertIn(token, multimodal_text)

    def test_program_map_and_pr_draft_describe_complete_rollout(self) -> None:
        program_map = (ROOT / "docs" / "00_program_map.md").read_text(encoding="utf-8")
        pr_draft = (
            ROOT
            / "docs"
            / "superpowers"
            / "prs"
            / "2026-03-31-btb-curriculum-redesign-pr.md"
        ).read_text(encoding="utf-8")

        self.assertIn("현재 rollout 상태", program_map)
        self.assertIn("02_nlp_bridge -> 03_nlp", program_map)
        self.assertIn("04_multimodal_bridge -> 05_multimodal", program_map)

        self.assertIn("00→05 foundations/bridge/applied", pr_draft)
        self.assertIn("### 5) bridge rollout 확장", pr_draft)
        self.assertIn("### 6) applied rollout 확장", pr_draft)
        self.assertNotIn("두 unit 모두", pr_draft)


if __name__ == "__main__":
    unittest.main()
