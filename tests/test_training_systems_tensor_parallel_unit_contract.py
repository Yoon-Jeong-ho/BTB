"""Contract tests for the runnable tensor-parallelism training-systems unit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "06_training_systems" / "05_tensor_parallelism"
ARTIFACTS = UNIT / "artifacts"


class TensorParallelUnitContractTest(unittest.TestCase):
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
        self.assertIn("python 06_training_systems/05_tensor_parallelism/scratch_lab.py", readme)
        self.assertIn("python 06_training_systems/05_tensor_parallelism/framework_lab.py", readme)
        self.assertIn("python 06_training_systems/05_tensor_parallelism/analysis.py", readme)
        self.assertNotIn("expected output / sample shape only", readme)
        self.assertNotIn("후속 applied", readme)

        for keyword in ["텐서 병렬", "행렬 shard", "activation shard", "communication overhead"]:
            self.assertIn(keyword, readme + theory)

        for keyword in ["FSDP", "pipeline", "row-parallel", "column-parallel"]:
            self.assertIn(keyword, readme + theory + prereqs + reflection)

    def test_lesson_metadata_marks_runnable_cpu_safe_unit(self) -> None:
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")

        for snippet in [
            "status: runnable",
            "unit: 05_tensor_parallelism",
            "track: 06_training_systems",
            "cpu_safe: true",
            "deterministic: true",
            "tensor_parallelism",
            "matrix_shard",
            "activation_shard",
            "communication_overhead",
        ]:
            self.assertIn(snippet, lesson)

    def test_scratch_lab_runs_and_writes_metrics_and_svg(self) -> None:
        result = self._run_json(UNIT / "scratch_lab.py")

        self.assertEqual("runnable", result["status"])
        self.assertEqual(4, result["tp_world_size"])
        self.assertEqual([3, 8], result["input_shape"])
        self.assertEqual([8, 16], result["column_parallel"]["global_weight_shape"])
        self.assertEqual([8, 4], result["column_parallel"]["per_rank_weight_shape"])
        self.assertEqual([3, 4], result["column_parallel"]["per_rank_activation_shape"])
        self.assertEqual([16, 6], result["row_parallel"]["global_weight_shape"])
        self.assertEqual([4, 6], result["row_parallel"]["per_rank_weight_shape"])
        self.assertEqual([3, 4], result["row_parallel"]["per_rank_activation_shape"])
        self.assertLess(result["max_abs_diff_vs_dense"], 1e-9)
        self.assertGreater(result["communication_overhead"]["estimated_bytes"], 0)

        metrics_path = ARTIFACTS / "scratch_metrics.json"
        svg_path = ARTIFACTS / "tensor_parallelism_shards.svg"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(svg_path.exists())
        self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))
        self.assertEqual(result, json.loads(metrics_path.read_text(encoding="utf-8")))

    def test_framework_lab_runs_deterministic_tp_simulation(self) -> None:
        first = self._run_json(UNIT / "framework_lab.py")
        second = self._run_json(UNIT / "framework_lab.py")

        self.assertEqual(first, second)
        self.assertEqual("runnable", first["status"])
        self.assertEqual("deterministic_cpu_tensor_parallel_sim", first["framework"])
        self.assertEqual(4, first["tp_world_size"])
        self.assertEqual(2, first["attention_partition"]["heads_per_rank"])
        self.assertEqual(["all_gather_activations", "all_reduce_partial_outputs"], first["collectives_per_block"])
        self.assertLess(first["numerical_check"]["max_abs_diff_vs_dense"], 1e-9)
        self.assertGreater(first["throughput_model"]["communication_share"], 0)
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
        self.assertIn("Tensor parallelism is an intra-layer split", stable)
        self.assertEqual(observed, observed_json)

        report = subprocess.run(
            [sys.executable, "scripts/build_lesson_report.py", "--unit", str(UNIT.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(0, report.returncode, report.stdout + report.stderr)
        summary = (ARTIFACTS / "summary.md").read_text(encoding="utf-8")
        self.assertIn("scratch_metrics.json", summary)
        self.assertIn("framework_metrics.json", summary)
        self.assertIn("analysis_observed.json", summary)

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
