from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumTrackDocs(unittest.TestCase):
    def test_learner_preflight_diagnoses_entry_skills_and_routes_gaps(self) -> None:
        path = ROOT / "docs" / "00_learner_preflight.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")

        for token in [
            "Python / CLI",
            "수학",
            "확률 / metric",
            "PyTorch / GPU",
            "진단 결과별 추천 경로",
            "00_foundations",
            "01_ml",
            "02_deep_learning",
        ]:
            self.assertIn(token, text)

        for doc in ["README.md", "docs/00_program_map.md", "docs/02_study_guide.md"]:
            doc_text = (ROOT / doc).read_text(encoding="utf-8")
            self.assertIn("docs/00_learner_preflight.md", doc_text)
            self.assertIn("선택형 사이드카", doc_text)

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


    def test_beginner_route_defers_systems_frontier_and_removes_stale_bridge_name(self) -> None:
        root = (ROOT / "README.md").read_text(encoding="utf-8")
        program_map = (ROOT / "docs" / "00_program_map.md").read_text(encoding="utf-8")
        guide = (ROOT / "docs" / "02_study_guide.md").read_text(encoding="utf-8")
        foundations = (ROOT / "00_foundations" / "README.md").read_text(encoding="utf-8")
        ml = (ROOT / "01_ml" / "README.md").read_text(encoding="utf-8")
        frontier = (ROOT / "07_frontier_labs" / "README.md").read_text(encoding="utf-8")

        for text in [root, program_map, guide]:
            self.assertIn("06_training_systems", text)
            self.assertIn("07_frontier_labs", text)
            self.assertRegex(text, r"(미뤄|나중|선택|optional|capstone sandbox)")

        self.assertNotIn("02_nlp_bridge", ml)
        self.assertIn("selected `02_deep_learning`", foundations)
        self.assertRegex(frontier, r"(선택|optional|capstone sandbox|고급)")
        self.assertIn("grounding entry point", root + guide)

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

    def test_gpu_plan_distinguishes_toy_capability_from_real_evidence(self) -> None:
        plan = (ROOT / "docs" / "04_gpu_conda_experiment_plan.md").read_text(encoding="utf-8")
        for token in [
            "BTB_DEVICE=cuda",
            "00_foundations/05_gpu_memory_runtime",
            "05_advanced_nlp_llm/04_instruction_tuning_and_sft",
            "09_multimodal/01_image_text_retrieval",
            "10_vla/01_vision_language_action_grounding",
            "artifact",
            "device",
        ]:
            self.assertIn(token, plan)
        self.assertNotIn("06_rlhf_and_reasoning_rl/framework_lab.py", plan)

    def test_study_guide_surfaces_learner_bridge_docs_for_llm_multimodal_and_vla(self) -> None:
        guide = (ROOT / "docs" / "02_study_guide.md").read_text(encoding="utf-8")
        vla = (ROOT / "10_vla" / "README.md").read_text(encoding="utf-8")
        multimodal_bridge = (ROOT / "08_multimodal_bridge" / "README.md").read_text(encoding="utf-8")
        multimodal = (ROOT / "09_multimodal" / "README.md").read_text(encoding="utf-8")
        llm = (ROOT / "05_advanced_nlp_llm" / "README.md").read_text(encoding="utf-8")
        app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

        bridge_docs = {
            "docs/06_decoder_generation_bridge.md": ["autoregressive", "temperature", "KV-cache"],
            "docs/07_multimodal_generation_bridge.md": [
                "cross-attention",
                "VQA",
                "grounding failure",
                "이미지 토큰",
                "soft token",
                "Gemma 4 12B",
                "encoder-free",
            ],
            "docs/08_rl_to_vla_bridge.md": ["MDP", "trajectory", "behavior cloning", "offline RL"],
        }
        for rel, tokens in bridge_docs.items():
            path = ROOT / rel
            self.assertTrue(path.exists(), rel)
            text = path.read_text(encoding="utf-8")
            for token in tokens:
                self.assertIn(token, text)
            self.assertIn(rel, guide)

        self.assertIn("../docs/06_decoder_generation_bridge.md", llm)
        self.assertIn("../docs/07_multimodal_generation_bridge.md", multimodal_bridge + multimodal)
        self.assertIn("../docs/08_rl_to_vla_bridge.md", vla)
        self.assertIn("../docs/06_decoder_generation_bridge.md", app)
        self.assertIn("../docs/07_multimodal_generation_bridge.md", app)
        self.assertIn("../docs/08_rl_to_vla_bridge.md", app)

    def test_multimodal_docs_clarify_image_tokens_and_encoder_free_vlm(self) -> None:
        bridge = (ROOT / "docs" / "07_multimodal_generation_bridge.md").read_text(encoding="utf-8")
        multimodal = (ROOT / "09_multimodal" / "README.md").read_text(encoding="utf-8")
        vqa_theory = (
            ROOT / "09_multimodal" / "03_visual_question_answering" / "THEORY.md"
        ).read_text(encoding="utf-8")

        for token in [
            "토큰이라는 말이 항상 discrete vocabulary ID를 뜻하지는 않는다",
            "attention이 처리하는 sequence element",
            "patch embedding + position embedding",
            "contextualized visual token",
            "soft image token",
            "discrete image token",
            "PaliGemma",
            "SigLIP",
            "Gemma 4 12B",
            "encoder-free",
            "single matrix multiplication",
            "vision encoder 없음",
            "<|vision_start|><|image_pad|><|vision_end|>",
            "image_token_id",
            "self.image_token * num_image_tokens",
            "같은 `<|image_pad|>` ID가 1280번 반복",
            "4~16384",
            "256~1280",
            "258 tokens",
            "sequence position",
            "손잡이(handle)",
        ]:
            self.assertIn(token, bridge)

        self.assertIn("이미지 토큰/soft token", multimodal)
        self.assertIn("encoder-free VLM", multimodal)
        self.assertIn("무거운 vision encoder를 거치지 않은 patch embedding", vqa_theory)


if __name__ == "__main__":
    unittest.main()
