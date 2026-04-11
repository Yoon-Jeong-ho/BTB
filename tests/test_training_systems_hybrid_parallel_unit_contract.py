"""Contract tests for the runnable hybrid-parallel-topology training-systems unit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "06_training_systems" / "08_hybrid_parallel_topologies"
ARTIFACTS = UNIT / "artifacts"
SCRATCH_METRICS = ARTIFACTS / "scratch_metrics.json"
FRAMEWORK_METRICS = ARTIFACTS / "framework_metrics.json"
SVG = ARTIFACTS / "hybrid_topology_mesh.svg"
OBSERVED = ARTIFACTS / "analysis_observed.json"


class HybridParallelTopologiesUnitContractTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._cleanup_generated_artifacts()

    def tearDown(self) -> None:
        self._cleanup_generated_artifacts()

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
        self.assertEqual("", (ARTIFACTS / ".gitkeep").read_text(encoding="utf-8"))

    def test_docs_are_korean_first_and_runnable(self) -> None:
        readme = (UNIT / "README.md").read_text(encoding="utf-8")
        theory = (UNIT / "THEORY.md").read_text(encoding="utf-8")
        prereqs = (UNIT / "PREREQS.md").read_text(encoding="utf-8")
        reflection = (UNIT / "reflection.md").read_text(encoding="utf-8")
        analysis = (UNIT / "analysis.md").read_text(encoding="utf-8")

        self.assertIn("> Status: runnable", readme)
        self.assertIn("python 06_training_systems/08_hybrid_parallel_topologies/scratch_lab.py", readme)
        self.assertIn("python 06_training_systems/08_hybrid_parallel_topologies/framework_lab.py", readme)
        self.assertIn("python 06_training_systems/08_hybrid_parallel_topologies/analysis.py", readme)
        self.assertNotIn("Status: outlined", readme)
        self.assertNotIn("후속 applied", readme)
        self.assertNotIn("expected output / sample shape only", readme)

        for doc in [readme, theory, prereqs, reflection, analysis]:
            self.assertRegex(doc, r"[가-힣]")

        combined = readme + theory + prereqs + reflection + analysis
        for keyword in [
            "data parallel",
            "tensor parallel",
            "pipeline parallel",
            "FSDP",
            "device mesh",
            "communication tradeoff",
            "bottleneck",
            "checkpoint",
        ]:
            self.assertIn(keyword, combined)

    def test_lesson_metadata_marks_runnable_cpu_safe_unit(self) -> None:
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")

        for snippet in [
            "status: runnable",
            "unit: 08_hybrid_parallel_topologies",
            "track: 06_training_systems",
            "cpu_safe: true",
            "deterministic: true",
            "scratch_lab.py",
            "framework_lab.py",
            "analysis.py",
            "hybrid parallel topology",
            "device mesh",
            "data parallel",
            "tensor parallelism",
            "pipeline parallelism",
            "FSDP",
            "communication tradeoff",
            "bottleneck reasoning",
            "checkpoint portability",
        ]:
            self.assertIn(snippet, lesson)

    def test_analysis_requires_metrics_with_actionable_failure(self) -> None:
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

    def test_scratch_lab_runs_and_writes_topology_metrics_and_svg(self) -> None:
        result = self._run_json(UNIT / "scratch_lab.py")

        self.assertEqual("runnable", result["status"])
        self.assertTrue(result["cpu_safe_simulation"])
        self.assertEqual("deterministic_cpu_hybrid_topology_planner", result["simulation"])
        self.assertEqual(64, result["hardware"]["total_gpus"])
        self.assertEqual("tp4_pp2_dp8_fsdp_hybrid", result["preferred_candidate"])
        self.assertEqual(
            ["data_parallel", "tensor_parallel", "pipeline_parallel", "fsdp_state_sharding"],
            list(result["parallel_axes"].keys()),
        )

        candidates = result["candidate_topologies"]
        self.assertGreaterEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(64, candidate["world_size"])
            self.assertTrue(candidate["topology_fit"])
            self.assertIn("DP", candidate["axis_product"])
            self.assertIn("TP", candidate["axis_product"])
            self.assertIn("PP", candidate["axis_product"])
            self.assertIn("memory_budget", candidate)
            self.assertIn("communication_budget", candidate)
            self.assertIn("bottleneck_reasoning", candidate)
            self.assertGreater(candidate["memory_budget"]["memory_margin_gb"], 0)
            for hotspot in ["fsdp_all_gather_reduce_scatter", "pipeline_activation_send_recv", "dp_gradient_sync"]:
                self.assertIn(hotspot, candidate["communication_budget"]["communication_hotspots"])

        preferred = next(candidate for candidate in candidates if candidate["name"] == result["preferred_candidate"])
        self.assertEqual(8, preferred["data_parallel"])
        self.assertEqual(4, preferred["tensor_parallel"])
        self.assertEqual(2, preferred["pipeline_parallel"])
        self.assertTrue(preferred["communication_budget"]["tensor_parallel_kept_intra_node"])
        self.assertIn("fast intra-node", preferred["bottleneck_reasoning"])

        self.assertTrue(SCRATCH_METRICS.exists())
        self.assertTrue(SVG.exists())
        self.assertIn("<svg", SVG.read_text(encoding="utf-8"))
        self.assertEqual(result, json.loads(SCRATCH_METRICS.read_text(encoding="utf-8")))

    def test_framework_lab_runs_deterministic_topology_scoring(self) -> None:
        first = self._run_json(UNIT / "framework_lab.py")
        second = self._run_json(UNIT / "framework_lab.py")

        self.assertEqual(first, second)
        self.assertEqual("runnable", first["status"])
        self.assertEqual("deterministic_cpu_hybrid_parallel_topology_sim", first["framework"])
        self.assertEqual(64, first["world_size"])
        self.assertEqual("tp4_pp2_dp8_fsdp_hybrid", first["preferred_candidate"])
        self.assertEqual(
            ["data_parallel", "tensor_parallel", "pipeline_parallel", "fsdp_state_sharding"],
            first["device_mesh_axes"],
        )
        self.assertEqual("dp_outer / pp_middle / tp_inner", first["rank_mesh_contract"]["rank_order"])
        self.assertIn("latency-sensitive", first["rank_mesh_contract"]["tp_inner_reason"])
        self.assertIn("checkpoint", first["bottleneck_reasoning"]["checkpoint_portability"])
        self.assertIn("tp_all_reduce", first["communication_tradeoffs"]["collectives_to_profile"])
        self.assertTrue(FRAMEWORK_METRICS.exists())

    def test_analysis_writes_stable_and_observed_reports(self) -> None:
        self._run_json(UNIT / "scratch_lab.py")
        self._run_json(UNIT / "framework_lab.py")
        stable_before = (UNIT / "analysis.md").read_text(encoding="utf-8")
        observed = self._run_json(UNIT / "analysis.py")

        self.assertEqual("runnable", observed["status"])
        self.assertEqual("analysis.md", observed["stable_report"])
        self.assertEqual("tp4_pp2_dp8_fsdp_hybrid", observed["preferred_candidate"])
        self.assertEqual("DP8 x TP4 x PP2", observed["axis_product"])
        self.assertIn("data_parallel", observed["parallel_axes"])
        self.assertIn("tensor_parallel", observed["parallel_axes"])
        self.assertIn("pipeline_parallel", observed["parallel_axes"])
        self.assertIn("fsdp_state_sharding", observed["parallel_axes"])
        self.assertIn("tp_all_reduce", observed["communication_hotspots"])
        self.assertIn("pipeline_activation_send_recv", observed["communication_hotspots"])
        self.assertIn("checkpoint", observed["topology_lesson"])
        self.assertTrue(OBSERVED.exists())
        self.assertEqual(observed, json.loads(OBSERVED.read_text(encoding="utf-8")))

        stable_after = (UNIT / "analysis.md").read_text(encoding="utf-8")
        self.assertEqual(stable_before, stable_after)
        self.assertNotIn(str(ROOT), stable_after)
        self.assertIn("## Stable interpretation", stable_after)
        self.assertIn("## Korean-first reading", stable_after)
        self.assertIn("Hybrid parallel topology planning", stable_after)

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

    def _cleanup_generated_artifacts(self) -> None:
        for path in [SCRATCH_METRICS, FRAMEWORK_METRICS, SVG, OBSERVED]:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
