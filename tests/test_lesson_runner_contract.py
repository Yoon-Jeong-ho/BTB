from __future__ import annotations

import json
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

    def test_metadata_loader_parses_shallow_mapping(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SCRIPTS_DIR)))

        from _lesson_metadata import load_lesson_metadata

        with tempfile.TemporaryDirectory() as tmp_dir:
            lesson_path = Path(tmp_dir) / "lesson.yaml"
            lesson_path.write_text(
                "\n".join(
                    [
                        "objective: nested metadata contract",
                        "scripts:",
                        "  scratch: scratch_lab.py",
                        "  framework: framework_lab.py",
                        "  analysis: analysis.py",
                    ]
                ),
                encoding="utf-8",
            )

            metadata = load_lesson_metadata(lesson_path)

        self.assertEqual(
            {
                "scratch": "scratch_lab.py",
                "framework": "framework_lab.py",
                "analysis": "analysis.py",
            },
            metadata["scripts"],
        )

    def test_metadata_loader_rejects_deeper_nested_mapping_with_location(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SCRIPTS_DIR)))

        from _lesson_metadata import load_lesson_metadata

        with tempfile.TemporaryDirectory() as tmp_dir:
            lesson_path = Path(tmp_dir) / "lesson.yaml"
            lesson_path.write_text(
                "scripts:\n  scratch: scratch_lab.py\n    path: nested.py\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, rf"{lesson_path}:3:.*indentation"):
                load_lesson_metadata(lesson_path)

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
        self.assertIn("selected_device=cpu", result.stdout)

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

    def test_runner_all_mode_propagates_cpu_device_in_pedagogical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "tmp_unit"
            unit_path.mkdir()
            unit_path.joinpath("lesson.yaml").write_text(
                "objective: runner all-mode contract\n",
                encoding="utf-8",
            )
            script = """\
import json
import os
from pathlib import Path

root = Path(__file__).parent
events = root / "events.jsonl"
with events.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "mode": Path(__file__).stem,
        "device": os.environ.get("BTB_DEVICE"),
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\\n")
print(f"entrypoint={Path(__file__).stem}")
"""
            for filename in ("scratch_lab.py", "framework_lab.py", "analysis.py"):
                unit_path.joinpath(filename).write_text(script, encoding="utf-8")

            result = self._run(
                str(RUN_LESSON),
                "--unit",
                str(unit_path),
                "--mode",
                "all",
                "--device",
                "cpu",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("selected_device=cpu", result.stdout)
            self.assertIn("completed_modes=scratch,framework,analysis", result.stdout)
            self.assertLess(result.stdout.index("selected_device=cpu"), result.stdout.index("entrypoint=scratch_lab"))
            events = [json.loads(line) for line in unit_path.joinpath("events.jsonl").read_text().splitlines()]
            self.assertEqual(["scratch_lab", "framework_lab", "analysis"], [event["mode"] for event in events])
            self.assertEqual(["cpu", "cpu", "cpu"], [event["device"] for event in events])
            self.assertEqual(["", "", ""], [event["visible"] for event in events])

    def test_runner_analysis_mode_has_actionable_missing_entrypoint_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "tmp_unit"
            unit_path.mkdir()
            unit_path.joinpath("lesson.yaml").write_text("objective: analysis mode\n", encoding="utf-8")

            result = self._run(
                str(RUN_LESSON),
                "--unit",
                str(unit_path),
                "--mode",
                "analysis",
                "--device",
                "cpu",
            )

            self.assertNotEqual(result.returncode, 0)
            error_text = result.stdout + result.stderr
            self.assertIn("analysis.py", error_text)
            self.assertIn("--mode analysis", error_text)

    def test_runner_all_mode_uses_stage_entrypoint_for_real_data_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "stage_unit"
            unit_path.mkdir()
            unit_path.joinpath("lesson.yaml").write_text("objective: stage runner contract\n", encoding="utf-8")
            stale = unit_path / "artifacts" / "old-run" / "stale.json"
            stale.parent.mkdir(parents=True)
            stale.write_text('{"old": true}', encoding="utf-8")
            unit_path.joinpath("run_stage.py").write_text(
                "from pathlib import Path\n"
                "import os\n"
                "out = Path(__file__).parent / 'artifacts' / 'stage-manual' / 'metrics.json'\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "out.write_text('{\"device\": \"' + os.environ['BTB_DEVICE'] + '\"}')\n"
                "print('entrypoint=run_stage')\n",
                encoding="utf-8",
            )

            result = self._run(
                str(RUN_LESSON),
                "--unit",
                str(unit_path),
                "--mode",
                "all",
                "--device",
                "cpu",
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("entrypoint=run_stage", result.stdout)
            self.assertIn("completed_modes=stage", result.stdout)
            self.assertIn("stage-manual/metrics.json", result.stdout)
            self.assertNotIn("stale.json", result.stdout)

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
            analysis_path = unit_path / "artifacts" / "analysis-manual" / "latest_report.md"
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

    def test_report_builder_resolves_observed_report_and_discloses_unknown_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "tmp_unit"
            framework_metrics = unit_path / "artifacts" / "framework-manual" / "metrics.json"
            observed_report = unit_path / "artifacts" / "analysis-manual" / "latest_report.md"
            framework_metrics.parent.mkdir(parents=True)
            observed_report.parent.mkdir(parents=True)
            unit_path.joinpath("lesson.yaml").write_text(
                "\n".join(
                    [
                        "objective: evidence-rich report",
                        "required_outputs:",
                        "  - framework metrics json",
                        "  - observed analysis report",
                        "  - custom qualitative checklist",
                        "analysis_questions:",
                        "  - loss가 감소한 근거는 무엇인가?",
                    ]
                ),
                encoding="utf-8",
            )
            framework_metrics.write_text('{"device": "cpu", "loss": 0.25}', encoding="utf-8")
            observed_report.write_text("# Observed\n\nloss decreased.\n", encoding="utf-8")

            result = self._run(str(BUILD_REPORT), "--unit", str(unit_path))

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = unit_path.joinpath("artifacts", "summary.md").read_text(encoding="utf-8")
            self.assertIn("device: cpu", summary)
            self.assertIn("loss: 0.25", summary)
            self.assertIn("framework-manual/metrics.json", summary)
            self.assertIn("analysis-manual/latest_report.md", summary)
            self.assertIn("unverified declarations", summary)
            self.assertIn("custom qualitative checklist", summary)
            self.assertIn("loss가 감소한 근거는 무엇인가?", summary)

    def test_report_builder_treats_path_like_output_as_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            unit_path = Path(tmp_dir) / "tmp_unit"
            unit_path.mkdir()
            unit_path.joinpath("lesson.yaml").write_text(
                "\n".join(
                    [
                        "objective: explicit artifact path",
                        "required_outputs:",
                        "  - artifacts/custom/evidence.json",
                    ]
                ),
                encoding="utf-8",
            )

            result = self._run(str(BUILD_REPORT), "--unit", str(unit_path))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(unit_path / "artifacts" / "custom" / "evidence.json"), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
