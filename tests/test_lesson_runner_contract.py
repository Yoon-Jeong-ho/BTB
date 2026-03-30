from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RUN_LESSON = SCRIPTS_DIR / "run_lesson.py"
BUILD_REPORT = SCRIPTS_DIR / "build_lesson_report.py"
TENSOR_UNIT = ROOT / "00_foundations" / "01_tensor_shapes"
GPU_UNIT = ROOT / "00_foundations" / "05_gpu_memory_runtime"
TENSOR_SCRATCH_METRICS = TENSOR_UNIT / "artifacts" / "scratch-manual" / "metrics.json"
GPU_FRAMEWORK_METRICS = GPU_UNIT / "artifacts" / "framework-manual" / "metrics.json"
SUMMARY_PATH = TENSOR_UNIT / "artifacts" / "summary.md"


class TestLessonRunnerContract(unittest.TestCase):
    maxDiff = None

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def _preserve_path(self, path: Path) -> None:
        existed = path.exists()
        original = path.read_bytes() if existed else None

        def cleanup() -> None:
            if existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                assert original is not None
                path.write_bytes(original)
            elif path.exists():
                path.unlink()

            current = path.parent
            while current != ROOT and current.exists() and not any(current.iterdir()):
                current.rmdir()
                current = current.parent

        self.addCleanup(cleanup)

    def test_metadata_loader_parses_constrained_lesson_schema(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SCRIPTS_DIR)))

        from _lesson_metadata import load_lesson_metadata

        with tempfile.TemporaryDirectory() as tmp_dir:
            lesson_path = Path(tmp_dir) / "lesson.yaml"
            lesson_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "objective: 예시 목표",
                        "prereqs:",
                        "  - 선행 1",
                        "  - 선행 2",
                        "key_terms:",
                        "  - tensor",
                        "notes: 자유 텍스트",
                    ]
                ),
                encoding="utf-8",
            )

            metadata = load_lesson_metadata(lesson_path)

        self.assertEqual("예시 목표", metadata["objective"])
        self.assertEqual(["선행 1", "선행 2"], metadata["prereqs"])
        self.assertEqual(["tensor"], metadata["key_terms"])
        self.assertEqual("자유 텍스트", metadata["notes"])

    def test_runner_executes_tensor_shapes_scratch(self) -> None:
        self._preserve_path(TENSOR_SCRATCH_METRICS)

        result = self._run(
            str(RUN_LESSON),
            "--unit",
            "00_foundations/01_tensor_shapes",
            "--mode",
            "scratch",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("unit=00_foundations/01_tensor_shapes", result.stdout)
        self.assertIn("mode=scratch", result.stdout)
        self.assertIn("scratch-manual", result.stdout)

    def test_runner_executes_gpu_memory_runtime_framework(self) -> None:
        self._preserve_path(GPU_FRAMEWORK_METRICS)

        result = self._run(
            str(RUN_LESSON),
            "--unit",
            "00_foundations/05_gpu_memory_runtime",
            "--mode",
            "framework",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("unit=00_foundations/05_gpu_memory_runtime", result.stdout)
        self.assertIn("mode=framework", result.stdout)
        self.assertIn("framework-manual", result.stdout)

    def test_report_builder_creates_summary_scaffold(self) -> None:
        self._preserve_path(SUMMARY_PATH)

        result = self._run(
            str(BUILD_REPORT),
            "--unit",
            "00_foundations/01_tensor_shapes",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(SUMMARY_PATH.exists(), "summary.md should be created")
        self.assertIn("summary.md", result.stdout)

        summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
        self.assertIn("# 01_tensor_shapes 요약", summary_text)
        self.assertIn("scratch keys:", summary_text)
        self.assertIn("framework keys:", summary_text)

    def test_report_builder_fails_with_actionable_missing_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "tmp_unit"
            scratch_metrics = unit_path / "artifacts" / "scratch-manual" / "metrics.json"
            framework_metrics = unit_path / "artifacts" / "framework-manual" / "metrics.json"
            analysis_path = unit_path / "analysis.md"
            summary_path = unit_path / "artifacts" / "summary.md"

            scratch_metrics.parent.mkdir(parents=True, exist_ok=True)
            unit_path.joinpath("lesson.yaml").write_text(
                "\n".join(
                    [
                        "objective: 임시 목표",
                        "required_outputs:",
                        "  - scratch metrics json",
                        "  - framework metrics json",
                        "  - analysis markdown",
                    ]
                ),
                encoding="utf-8",
            )
            scratch_metrics.write_text('{"ok": true}', encoding="utf-8")

            result = self._run(str(BUILD_REPORT), "--unit", str(unit_path))

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(summary_path.exists(), "summary.md should not be created on failure")
            error_text = result.stdout + result.stderr
            self.assertIn("필수 출력이 없습니다", error_text)
            self.assertIn(str(framework_metrics), error_text)
            self.assertIn(str(analysis_path), error_text)
            self.assertIn("analysis.py", error_text)


if __name__ == "__main__":
    unittest.main()
