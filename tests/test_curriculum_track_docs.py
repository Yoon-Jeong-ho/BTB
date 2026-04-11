from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumTrackDocs(unittest.TestCase):
    def test_root_readme_removes_placeholder_language(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")

        for stale in ["재배치 대상", "자리를 미리 고정", "단계적으로 재배치 중", "향후 학습/평가"]:
            self.assertNotIn(stale, text)

        self.assertIn("foundations/deep-learning core/bridge/applied/systems/frontier/multimodal", text)
        self.assertIn("03_nlp_bridge -> 04_nlp", text)
        self.assertIn("08_multimodal_bridge -> 09_multimodal", text)
        self.assertIn("docs/02_study_guide.md", text)

    def test_reindexed_track_headings_match_directory_numbers(self) -> None:
        self.assertEqual(
            (ROOT / "04_nlp" / "README.md").read_text(encoding="utf-8").splitlines()[0],
            "# 04 NLP",
        )
        self.assertEqual(
            (ROOT / "09_multimodal" / "README.md").read_text(encoding="utf-8").splitlines()[0],
            "# 09 Multimodal",
        )

    def test_track_readmes_describe_real_populated_units(self) -> None:
        nlp_text = (ROOT / "04_nlp" / "README.md").read_text(encoding="utf-8")
        multimodal_text = (ROOT / "09_multimodal" / "README.md").read_text(encoding="utf-8")

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

        self.assertIn("현재 학습 가능 상태", program_map)
        self.assertIn("03_nlp_bridge -> 04_nlp", program_map)
        self.assertIn("08_multimodal_bridge -> 09_multimodal", program_map)
        self.assertIn("02_study_guide.md", program_map)

        self.assertIn("00→05 foundations/bridge/applied", pr_draft)
        self.assertIn("### 5) bridge rollout 확장", pr_draft)
        self.assertIn("### 6) applied rollout 확장", pr_draft)
        self.assertNotIn("두 unit 모두", pr_draft)

    def test_study_guide_surfaces_deep_learning_core_path(self) -> None:
        guide = (ROOT / "docs" / "02_study_guide.md").read_text(encoding="utf-8")

        self.assertIn("딥러닝 코어", guide)
        for rel in [
            "00_foundations/02_activation_and_loss",
            "00_foundations/03_gradients_and_backpropagation",
            "02_deep_learning/04_attention_and_transformers",
            "03_nlp_bridge/02_attention_and_transformer_block",
        ]:
            self.assertIn(rel, guide)


if __name__ == "__main__":
    unittest.main()
