from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "00_shared" / "templates"


class TestSharedTemplates(unittest.TestCase):
    def test_foundation_templates_have_required_markers(self) -> None:
        expected_markers = {
            "foundation_readme_template.md": [
                "## 왜 이 단위를 배우는가",
                "## 이번 단위에서 남길 것",
            ],
            "foundation_theory_template.md": [
                "## 핵심 개념",
                "## 수식 / 직관",
            ],
            "foundation_analysis_template.md": [
                "## 관측 결과",
                "## 해석",
            ],
            "foundation_reflection_template.md": [
                "이번에 이해한 것",
                "아직 애매한 것",
            ],
        }

        for name, markers in expected_markers.items():
            path = TEMPLATE_ROOT / name
            with self.subTest(template=name):
                self.assertTrue(path.exists(), name)
                text = path.read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, text)

    def test_shared_readme_mentions_unit_contract(self) -> None:
        text = (ROOT / "00_shared" / "README.md").read_text(encoding="utf-8")
        self.assertIn("README/THEORY/PREREQS/scratch/framework/analysis/reflection", text)
        for name in [
            "README.md",
            "THEORY.md",
            "PREREQS.md",
            "scratch_lab.py",
            "framework_lab.py",
            "analysis.md",
            "reflection.md",
        ]:
            self.assertIn(name, text)

    def test_playbook_mentions_explicit_unit_contract_phrases(self) -> None:
        text = (ROOT / "docs" / "01_experiment_playbook.md").read_text(encoding="utf-8")
        for needle in [
            "## 2. 모든 실험/학습 단위가 남겨야 하는 파일",
            "### 학습 단위 contract",
            "3절부터는 run/report artifact 승격 규약으로 읽는다.",
            "lesson.yaml",
            "analysis.md",
            "reflection.md",
            "출력 계약, 분석 질문",
            "결과 해설 문서",
            "학습자 관점 회고",
            "runtime 관련 실습은 GPU/CPU 관측치를 함께 남긴다.",
            "runtime observations는 숫자만 적지 말고 원인 해석까지 붙인다.",
        ]:
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
