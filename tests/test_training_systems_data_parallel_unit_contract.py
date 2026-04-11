"""Contract tests for the runnable data-parallel + gradient-accumulation unit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "06_training_systems" / "07_data_parallel_grad_accumulation"
ARTIFACTS = UNIT / "artifacts"


class DataParallelGradAccumUnitContractTest(unittest.TestCase):
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
        self.assertIn("python 06_training_systems/07_data_parallel_grad_accumulation/scratch_lab.py", readme)
        self.assertIn("python 06_training_systems/07_data_parallel_grad_accumulation/framework_lab.py", readme)
        self.assertIn("python 06_training_systems/07_data_parallel_grad_accumulation/analysis.py", readme)
        self.assertNotIn("expected output / sample shape only", readme)
        self.assertNotIn("후속 applied", readme)
        self.assertNotIn("Status: outlined", readme)

        combined = readme + theory + prereqs + reflection
        for keyword in [
            "data parallel",
            "grad accumulation",
            "local batch",
            "global batch",
            "effective batch",
            "optimizer step cadence",
            "loss normalization",
            "gradient clipping",
            "deferred sync",
            "no_sync",
            "all-reduce",
        ]:
            self.assertIn(keyword, combined)
        for doc in [readme, theory, prereqs, reflection]:
            self.assertRegex(doc, r"[가-힣]")

    def test_lesson_metadata_marks_runnable_cpu_safe_unit(self) -> None:
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")

        for snippet in [
            "status: runnable",
            "unit: 07_data_parallel_grad_accumulation",
            "track: 06_training_systems",
            "cpu_safe: true",
            "deterministic: true",
            "data_parallel",
            "gradient_accumulation",
            "effective_batch",
            "optimizer_step_cadence",
            "deferred_sync",
            "loss_normalization",
        ]:
            self.assertIn(snippet, lesson)

    def test_scratch_lab_runs_and_writes_metrics_and_svg(self) -> None:
        result = self._run_json(UNIT / "scratch_lab.py")

        self.assertEqual("runnable", result["status"])
        self.assertTrue(result["cpu_safe_simulation"])
        self.assertEqual(4, result["world_size"])
        self.assertEqual(8, result["local_batch_size"])
        self.assertEqual(4, result["grad_accum_steps"])
        self.assertEqual(32, result["global_batch_per_microstep"])
        self.assertEqual(128, result["effective_batch_per_optimizer_step"])
        self.assertEqual(8, result["microstep_count"])
        self.assertEqual(2, result["optimizer_step_count"])
        self.assertEqual(8, result["sync_policy_comparison"]["every_step_all_reduce_count"])
        self.assertEqual(2, result["sync_policy_comparison"]["deferred_sync_all_reduce_count"])
        self.assertLess(
            result["sync_policy_comparison"]["deferred_sync_all_reduce_count"],
            result["sync_policy_comparison"]["every_step_all_reduce_count"],
        )
        self.assertAlmostEqual(0.25, result["loss_normalization"]["scale_per_microstep"])
        self.assertEqual("clip_after_accumulation_boundary", result["gradient_clipping"]["recommended_timing"])
        self.assertLess(
            result["memory_model_mb"]["small_local_batch_with_accumulation_peak"],
            result["memory_model_mb"]["equivalent_large_local_batch_peak"],
        )
        self.assertEqual("all_reduce_gradients", result["accumulation_trace"][3]["collective"])
        self.assertTrue(result["accumulation_trace"][3]["optimizer_step"])
        self.assertFalse(result["accumulation_trace"][2]["sync_gradients"])

        metrics_path = ARTIFACTS / "scratch_metrics.json"
        svg_path = ARTIFACTS / "data_parallel_grad_accumulation.svg"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(svg_path.exists())
        self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))
        self.assertEqual(result, json.loads(metrics_path.read_text(encoding="utf-8")))

    def test_framework_lab_runs_deterministic_accumulation_simulation(self) -> None:
        first = self._run_json(UNIT / "framework_lab.py")
        second = self._run_json(UNIT / "framework_lab.py")

        self.assertEqual(first, second)
        self.assertEqual("runnable", first["status"])
        self.assertEqual("deterministic_cpu_data_parallel_grad_accum_sim", first["framework"])
        self.assertEqual("cpu_fallback", first["backend"])
        self.assertEqual(4, first["rank_count"])
        self.assertEqual(3, first["accumulation_steps"])
        self.assertEqual(24, first["global_batch_per_microstep"])
        self.assertEqual(72, first["effective_batch_per_optimizer_step"])
        self.assertEqual(3, first["optimizer_step_cadence"]["microsteps_per_optimizer_step"])
        self.assertEqual(2, first["optimizer_step_cadence"]["optimizer_steps"])
        self.assertIn("local_backward_no_sync", first["collectives"])
        self.assertIn("boundary_all_reduce_gradients", first["collectives"])
        self.assertIn("optimizer_step", first["collectives"])
        self.assertLess(first["communication_model"]["deferred_sync_calls"], first["communication_model"]["every_step_sync_calls"])
        self.assertGreater(first["throughput_model"]["tokens_per_optimizer_step"], first["throughput_model"]["tokens_per_microstep"])
        for rank_window in first["rank_windows"]:
            self.assertEqual(384.0, rank_window["gradient_buffer_mb"])
            self.assertEqual("boundary_all_reduce_gradients", rank_window["accumulation_slots"][2]["operation"])
            self.assertIn("not sharded in DDP", rank_window["replica_contract"])
        self.assertEqual("scheduler_steps_on_optimizer_step", first["scheduler_policy"])
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
        self.assertEqual(128, observed["scratch_effective_batch"])
        self.assertEqual(72, observed["framework_effective_batch"])
        stable = (UNIT / "analysis.md").read_text(encoding="utf-8")
        observed_json = json.loads((ARTIFACTS / "analysis_observed.json").read_text(encoding="utf-8"))

        self.assertNotIn(str(ROOT), stable)
        self.assertIn("## Stable interpretation", stable)
        self.assertIn("## Korean-first reading", stable)
        self.assertIn("Data parallelism expands the batch axis", stable)
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
