from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "10_vla"
UNIT = TRACK / "01_vision_language_action_grounding"
ARTIFACTS = UNIT / "artifacts"
SCRATCH_DIR = ARTIFACTS / "scratch-manual"
FRAMEWORK_DIR = ARTIFACTS / "framework-manual"
ANALYSIS_DIR = ARTIFACTS / "analysis-manual"
SCRATCH_METRICS = SCRATCH_DIR / "metrics.json"
SCRATCH_FIGURE = SCRATCH_DIR / "action_policy_matrix.svg"
FRAMEWORK_METRICS = FRAMEWORK_DIR / "metrics.json"
OBSERVED_REPORT = ANALYSIS_DIR / "latest_report.md"
FAILURE_PROBE_LABELS = {
    "wrong action but safe",
    "right action but unsafe",
    "ambiguous instruction",
    "observation noise",
}


class VLAUnitContractTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._cleanup_generated_outputs()

    def tearDown(self) -> None:
        self._cleanup_generated_outputs()

    def _cleanup_generated_outputs(self) -> None:
        for directory in [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]:
            if directory.exists():
                shutil.rmtree(directory)

    def _run(self, relative_path: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )

    def _run_json(self, relative_path: str) -> dict[str, object]:
        result = self._run(relative_path)
        self.assertEqual(0, result.returncode, result.stderr)
        return json.loads(result.stdout)

    def test_required_files_and_korean_docs_exist(self) -> None:
        required = [
            TRACK / "README.md",
            UNIT / "README.md",
            UNIT / "THEORY.md",
            UNIT / "PREREQS.md",
            UNIT / "lesson.yaml",
            UNIT / "scratch_lab.py",
            UNIT / "framework_lab.py",
            UNIT / "analysis.py",
            UNIT / "analysis.md",
            UNIT / "reflection.md",
            ARTIFACTS / ".gitkeep",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        self.assertEqual([], missing)
        self.assertEqual("", (ARTIFACTS / ".gitkeep").read_text(encoding="utf-8"))

        readme = (UNIT / "README.md").read_text(encoding="utf-8")
        theory = (UNIT / "THEORY.md").read_text(encoding="utf-8")
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")
        self.assertRegex(readme, r"[가-힣]")
        self.assertRegex(theory, r"[가-힣]")
        self.assertIn("Status: runnable", readme)
        self.assertIn("실행 결과 예시", readme)
        self.assertIn("VLA", readme)
        self.assertIn("action token", theory)
        self.assertIn("wrong action but safe", readme + theory)
        self.assertIn("right action but unsafe", readme + theory)
        self.assertIn("ambiguous instruction", readme + theory)
        self.assertIn("observation noise", readme + theory)
        self.assertIn("status: runnable", lesson)
        self.assertIn("vision-language-action", lesson)
        self.assertIn("safety gate", lesson)

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        result = self._run("10_vla/01_vision_language_action_grounding/analysis.py")
        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn("필수 VLA metrics 파일이 없습니다", error_text)
        self.assertIn("먼저 scratch_lab.py와 framework_lab.py를 실행하세요", error_text)

    def test_framework_rejects_unknown_btb_device(self) -> None:
        env = {**os.environ, "BTB_DEVICE": "invalid"}

        result = self._run(
            "10_vla/01_vision_language_action_grounding/framework_lab.py",
            env=env,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("auto, cpu, or cuda", result.stdout + result.stderr)

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        scratch = self._run_json("10_vla/01_vision_language_action_grounding/scratch_lab.py")
        framework = self._run_json("10_vla/01_vision_language_action_grounding/framework_lab.py")
        analysis = self._run_json("10_vla/01_vision_language_action_grounding/analysis.py")

        self.assertEqual("vision_language_action_grounding", scratch["unit"])
        self.assertEqual(4, scratch["scenario_count"])
        self.assertEqual(1.0, scratch["action_accuracy"])
        self.assertEqual(1.0, scratch["safety_gate_accuracy"])
        self.assertEqual([4, 4], scratch["policy_matrix_shape"])
        self.assertEqual("artifacts/scratch-manual/action_policy_matrix.svg", scratch["figure_path"])
        self.assertEqual(FAILURE_PROBE_LABELS, set(scratch["failure_probe_labels"]))
        self.assertEqual(4, scratch["failure_probe_counts"]["total"])
        self.assertEqual(FAILURE_PROBE_LABELS, set(scratch["failure_probe_counts"]["by_label"]))
        self.assertEqual(4, len(scratch["failure_probe_rows"]))
        for row in scratch["failure_probe_rows"]:
            self.assertIn(row["probe_label"], FAILURE_PROBE_LABELS)
            self.assertIn("expected_behavior", row)
            self.assertIn("observed_behavior", row)
            self.assertIn("probe_passed", row)
        self.assertTrue(SCRATCH_METRICS.exists())
        self.assertTrue(SCRATCH_FIGURE.exists())
        self.assertIn("<svg", SCRATCH_FIGURE.read_text(encoding="utf-8"))
        self.assertIn("VLA action policy matrix", SCRATCH_FIGURE.read_text(encoding="utf-8"))

        self.assertEqual("vision_language_action_grounding", framework["unit"])
        self.assertIn(framework["device"], {"cpu", "cuda"})
        self.assertEqual([4, 4], framework["logits_shape"])
        self.assertEqual(1.0, framework["action_accuracy"])
        self.assertEqual(FAILURE_PROBE_LABELS, set(framework["failure_probe_labels"]))
        self.assertEqual(4, framework["failure_probe_counts"]["total"])
        self.assertEqual(FAILURE_PROBE_LABELS, set(framework["failure_probe_counts"]["by_label"]))
        self.assertEqual(4, len(framework["failure_probe_rows"]))
        self.assertLess(framework["loss_history_tail"][-1], framework["loss_history_head"][0])
        self.assertTrue(FRAMEWORK_METRICS.exists())

        self.assertEqual("vision_language_action_grounding", analysis["unit"])
        self.assertEqual("runnable", analysis["status"])
        self.assertEqual(1.0, analysis["scratch_action_accuracy"])
        self.assertEqual(1.0, analysis["framework_action_accuracy"])
        self.assertEqual(4, analysis["scratch_failure_probe_counts"]["total"])
        self.assertEqual(4, analysis["framework_failure_probe_counts"]["total"])
        self.assertTrue(OBSERVED_REPORT.exists())
        report = OBSERVED_REPORT.read_text(encoding="utf-8")
        self.assertIn("# 01 Vision-Language-Action Grounding 실행 관측", report)
        self.assertIn("## 한국어 해석", report)
        self.assertIn("safety gate", report)
        self.assertIn("## 실패 probe 관측", report)
        for label in FAILURE_PROBE_LABELS:
            self.assertIn(label, report)
        self.assertNotIn(str(ROOT), report)


if __name__ == "__main__":
    unittest.main()
