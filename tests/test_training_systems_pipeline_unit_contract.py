"""Contract tests for the runnable pipeline-parallelism training-systems unit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "06_training_systems" / "06_pipeline_parallelism"
ARTIFACTS = UNIT / "artifacts"


class PipelineParallelUnitContractTest(unittest.TestCase):
    maxDiff = None

    def test_required_files_exist(self) -> None:
        required = [
            "README.md",
            "THEORY.md",
            "PREREQS.md",
            "lesson.yaml",
            "scratch_lab.py",
            "framework_lab.py",
            "analysis.py",
            "analysis.md",
            "reflection.md",
            "artifacts/.gitkeep",
        ]

        missing = [relative for relative in required if not (UNIT / relative).exists()]

        self.assertEqual([], missing)

    def test_docs_are_korean_first_and_runnable(self) -> None:
        readme = (UNIT / "README.md").read_text(encoding="utf-8")
        theory = (UNIT / "THEORY.md").read_text(encoding="utf-8")
        prereqs = (UNIT / "PREREQS.md").read_text(encoding="utf-8")
        reflection = (UNIT / "reflection.md").read_text(encoding="utf-8")

        self.assertIn("> Status: runnable", readme)
        self.assertIn("python 06_training_systems/06_pipeline_parallelism/scratch_lab.py", readme)
        self.assertIn("python 06_training_systems/06_pipeline_parallelism/framework_lab.py", readme)
        self.assertIn("python 06_training_systems/06_pipeline_parallelism/analysis.py", readme)
        self.assertNotIn("expected output / sample shape only", readme)
        self.assertNotIn("후속 applied", readme)

        combined = readme + theory + prereqs + reflection
        for keyword in [
            "pipeline stage",
            "microbatch",
            "bubble",
            "throughput",
            "activation transfer",
            "partition",
            "1F1B",
        ]:
            self.assertIn(keyword, combined)

    def test_lesson_metadata_marks_runnable_cpu_safe_unit(self) -> None:
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")

        for snippet in [
            "status: runnable",
            "unit: 06_pipeline_parallelism",
            "track: 06_training_systems",
            "cpu_safe: true",
            "deterministic: true",
            "pipeline_stage",
            "microbatch_schedule",
            "bubble_fraction",
            "activation_transfer",
            "partition_balance",
        ]:
            self.assertIn(snippet, lesson)

    def test_scratch_lab_runs_and_writes_metrics_and_svg(self) -> None:
        result = self._run_json(UNIT / "scratch_lab.py")

        self.assertEqual("runnable", result["status"])
        self.assertEqual("deterministic_cpu_pipeline_schedule", result["simulation"])
        self.assertEqual(3, result["num_stages"])
        self.assertEqual(6, result["microbatches"])
        self.assertEqual("forward_pipeline_fill_drain", result["schedule_summary"]["policy"])
        self.assertEqual(8, result["schedule_summary"]["total_time_slots"])
        self.assertEqual(6, result["schedule_summary"]["idle_stage_slots"])
        self.assertAlmostEqual(0.25, result["schedule_summary"]["bubble_fraction"])
        self.assertGreater(result["activation_transfer"]["estimated_bytes"], 0)
        self.assertGreater(result["partition_balance"]["max_over_min_stage_compute"], 1.0)

        metrics_path = ARTIFACTS / "scratch_metrics.json"
        svg_path = ARTIFACTS / "pipeline_schedule.svg"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(svg_path.exists())
        self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))
        self.assertEqual(result, json.loads(metrics_path.read_text(encoding="utf-8")))

    def test_framework_lab_runs_deterministic_pipeline_simulation(self) -> None:
        first = self._run_json(UNIT / "framework_lab.py")
        second = self._run_json(UNIT / "framework_lab.py")

        self.assertEqual(first, second)
        self.assertEqual("runnable", first["status"])
        self.assertEqual("deterministic_cpu_pipeline_parallel_sim", first["framework"])
        self.assertEqual(4, first["num_stages"])
        self.assertEqual(8, first["microbatches"])
        self.assertEqual("1F1B_greedy_dependency_sim", first["schedule_policy"])
        self.assertEqual(["forward_activation_send", "backward_gradient_recv"], first["transfers_per_boundary"])
        self.assertGreater(first["schedule_metrics"]["bubble_fraction"], 0)
        self.assertGreater(first["activation_memory_model"]["gpipe_peak_saved_microbatches"], first["activation_memory_model"]["one_f1b_peak_saved_microbatches"])
        self.assertTrue((ARTIFACTS / "framework_metrics.json").exists())

    def test_analysis_requires_metrics_and_writes_stable_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            failed = subprocess.run(
                [
                    sys.executable,
                    str(UNIT / "analysis.py"),
                    "--scratch-metrics",
                    str(Path(tmpdir) / "missing_scratch.json"),
                    "--framework-metrics",
                    str(Path(tmpdir) / "missing_framework.json"),
                    "--output",
                    str(Path(tmpdir) / "analysis.md"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(0, failed.returncode)
        self.assertIn("Missing required metrics file", failed.stderr)
        self.assertIn("Run scratch_lab.py and framework_lab.py first", failed.stderr)

        self._run_json(UNIT / "scratch_lab.py")
        self._run_json(UNIT / "framework_lab.py")
        observed = self._run_json(UNIT / "analysis.py")

        self.assertEqual("runnable", observed["status"])
        self.assertEqual("analysis.md", observed["stable_report"])
        self.assertTrue(observed["observed_report"].endswith("analysis_observed.json"))
        stable = (UNIT / "analysis.md").read_text(encoding="utf-8")
        observed_json = json.loads((ARTIFACTS / "analysis_observed.json").read_text(encoding="utf-8"))

        self.assertNotIn(str(ROOT), stable)
        self.assertIn("## Stable interpretation", stable)
        self.assertIn("## Observed run", stable)
        self.assertIn("Pipeline parallelism is execution-path partitioning", stable)
        self.assertEqual(observed, observed_json)

    def _run_json(self, script: Path) -> dict:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostics for failures
            raise AssertionError(
                f"{script.name} did not emit JSON.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            ) from exc


if __name__ == "__main__":
    unittest.main()
