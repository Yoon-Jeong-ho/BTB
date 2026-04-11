"""Contract tests for the runnable frontier capstone model-building unit."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "07_frontier_labs" / "02_capstone_model_building"
ARTIFACTS = UNIT / "artifacts"
SCRATCH_DIR = ARTIFACTS / "scratch-manual"
FRAMEWORK_DIR = ARTIFACTS / "framework-manual"
ANALYSIS_DIR = ARTIFACTS / "analysis-manual"
SCRATCH_CONTRACT = SCRATCH_DIR / "capstone_contract.json"
SCRATCH_FIGURE = SCRATCH_DIR / "milestone_gates.svg"
FRAMEWORK_BOARD = FRAMEWORK_DIR / "project_board.json"
OBSERVED_REPORT = ANALYSIS_DIR / "latest_report.md"


class FrontierLabsCapstoneUnitContractTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._cleanup_generated_outputs()

    def tearDown(self) -> None:
        self._cleanup_generated_outputs()

    def test_required_runnable_files_exist(self) -> None:
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

    def test_docs_and_metadata_are_korean_first_runnable_contract(self) -> None:
        readme = (UNIT / "README.md").read_text(encoding="utf-8")
        analysis = (UNIT / "analysis.md").read_text(encoding="utf-8")
        reflection = (UNIT / "reflection.md").read_text(encoding="utf-8")
        lesson = (UNIT / "lesson.yaml").read_text(encoding="utf-8")

        self.assertIn("> Status: runnable", readme)
        self.assertIn("CPU-safe deterministic", readme)
        self.assertIn("python3 07_frontier_labs/02_capstone_model_building/scratch_lab.py", readme)
        self.assertIn("python3 07_frontier_labs/02_capstone_model_building/framework_lab.py", readme)
        self.assertIn("python3 07_frontier_labs/02_capstone_model_building/analysis.py", readme)
        self.assertNotIn("Status: outlined", readme)
        self.assertNotIn("후속 applied", readme)
        self.assertNotIn("expected output / sample shape only", readme)

        for doc in [readme, analysis, reflection]:
            self.assertRegex(doc, r"[가-힣]")

        for snippet in [
            "status: runnable",
            "unit: 02_capstone_model_building",
            "track: 07_frontier_labs",
            "cpu_safe: true",
            "deterministic: true",
            "scratch_lab.py",
            "framework_lab.py",
            "analysis.py",
            "problem statement",
            "non-goal",
            "dataset contract",
            "model contract",
            "eval contract",
            "acceptance gate",
            "risk register",
            "failure analysis",
            "report outline",
        ]:
            self.assertIn(snippet, lesson)

    def test_scratch_lab_writes_deterministic_capstone_contract(self) -> None:
        first = self._run_json("07_frontier_labs/02_capstone_model_building/scratch_lab.py")
        second = self._run_json("07_frontier_labs/02_capstone_model_building/scratch_lab.py")

        self.assertEqual(first, second)
        self.assertEqual("runnable", first["status"])
        self.assertTrue(first["cpu_safe_simulation"])
        self.assertEqual("capstone_model_building_contract", first["contract_type"])
        self.assertIn("Recall@10", first["problem_statement"])
        self.assertGreaterEqual(len(first["non_goals"]), 3)
        self.assertIn("real-time serving", " ".join(first["non_goals"]))

        dataset = first["dataset_contract"]
        self.assertEqual("synthetic_korean_catalog_seed_v1", dataset["source"])
        self.assertEqual({"train": 1200, "valid": 200, "test": 200}, dataset["split"])
        self.assertIn("query", dataset["schema_fields"])
        self.assertIn("image_caption", dataset["schema_fields"])
        self.assertIn("near_duplicate_group_holdout", dataset["leakage_controls"])

        model = first["model_contract"]
        self.assertEqual("lexical_title_baseline", model["baseline"])
        self.assertIn("tiny_dual_encoder", model["candidates"])
        self.assertEqual("cpu toy embedding table", model["runtime_budget"])

        eval_contract = first["eval_contract"]
        self.assertEqual("Recall@10", eval_contract["primary_metric"])
        self.assertEqual(0.42, eval_contract["baseline_score"])
        self.assertEqual(0.49, eval_contract["target_score"])
        self.assertEqual(0.05, eval_contract["minimum_delta"])
        self.assertIn("brand_mismatch", eval_contract["qualitative_buckets"])

        milestone_ids = [milestone["id"] for milestone in first["milestones"]]
        self.assertEqual(["M0", "M1", "M2", "M3"], milestone_ids)
        for milestone in first["milestones"]:
            self.assertIn("acceptance_gate", milestone)
            self.assertIn("required_artifacts", milestone)

        self.assertGreaterEqual(len(first["risk_register"]), 5)
        self.assertIn("dataset_leakage", {risk["id"] for risk in first["risk_register"]})
        self.assertEqual(
            ["slice", "failure_bucket", "evidence", "hypothesis", "next_action"],
            first["failure_analysis_outline"]["columns"],
        )
        self.assertTrue(SCRATCH_CONTRACT.exists())
        self.assertTrue(SCRATCH_FIGURE.exists())
        self.assertIn("<svg", SCRATCH_FIGURE.read_text(encoding="utf-8"))
        self.assertEqual(first, json.loads(SCRATCH_CONTRACT.read_text(encoding="utf-8")))

    def test_framework_lab_writes_project_board_and_acceptance_gates(self) -> None:
        result = self._run_json("07_frontier_labs/02_capstone_model_building/framework_lab.py")

        self.assertEqual("runnable", result["status"])
        self.assertEqual("cpu_capstone_project_board_sim", result["framework"])
        self.assertTrue(result["cpu_safe_simulation"])
        self.assertEqual("korean_catalog_retrieval_capstone", result["project_id"])
        self.assertEqual("problem_scope_frozen", result["acceptance_gate_verdicts"][0]["gate"])
        self.assertEqual("pass", result["acceptance_gate_verdicts"][0]["verdict"])
        self.assertEqual("blocked_until_artifacts_complete", result["acceptance_gate_verdicts"][-1]["verdict"])

        matrix = result["dataset_model_eval_matrix"]
        self.assertEqual("fixed synthetic split", matrix["dataset"])
        self.assertEqual("lexical baseline vs tiny dual encoder", matrix["model_comparison"])
        self.assertEqual("Recall@10 + slice review", matrix["eval_protocol"])

        board = result["milestone_board"]
        self.assertEqual(["M0", "M1", "M2", "M3"], [card["id"] for card in board])
        self.assertEqual("done", board[0]["state"])
        self.assertIn("final_capstone_report.md", board[-1]["exit_artifacts"])

        report_sections = result["report_outline"]
        self.assertEqual("problem_and_non_goals", report_sections[0])
        self.assertIn("failure_analysis", report_sections)
        self.assertIn("next_steps", report_sections)
        self.assertIn("retry budget", result["handoff_to_agentic_loop"]["stop_rules"][0])
        self.assertIn("scope_creep", {risk["id"] for risk in result["risk_register"]})
        self.assertTrue(FRAMEWORK_BOARD.exists())
        self.assertEqual(result, json.loads(FRAMEWORK_BOARD.read_text(encoding="utf-8")))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            failed = subprocess.run(
                [
                    sys.executable,
                    "07_frontier_labs/02_capstone_model_building/analysis.py",
                    "--scratch-contract",
                    str(Path(tmpdir) / "missing_contract.json"),
                    "--framework-board",
                    str(Path(tmpdir) / "missing_board.json"),
                    "--output",
                    str(Path(tmpdir) / "latest_report.md"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(0, failed.returncode)
        error_text = failed.stdout + failed.stderr
        self.assertIn("Missing required capstone artifact", error_text)
        self.assertIn("Run scratch_lab.py and framework_lab.py first", error_text)

    def test_analysis_writes_failure_report_outline_from_artifacts(self) -> None:
        self._run_json("07_frontier_labs/02_capstone_model_building/scratch_lab.py")
        self._run_json("07_frontier_labs/02_capstone_model_building/framework_lab.py")
        observed = self._run_json("07_frontier_labs/02_capstone_model_building/analysis.py")

        self.assertEqual("runnable", observed["status"])
        self.assertEqual("korean_catalog_retrieval_capstone", observed["project_id"])
        self.assertEqual("Recall@10", observed["primary_metric"])
        self.assertEqual(0.07, observed["target_delta"])
        self.assertEqual(["M0", "M1", "M2", "M3"], observed["milestone_ids"])
        self.assertIn("dataset_leakage", observed["top_risks"])
        self.assertIn("failure_analysis", observed["report_outline"])
        self.assertEqual("blocked_until_artifacts_complete", observed["final_gate_verdict"])
        self.assertTrue(OBSERVED_REPORT.exists())
        report = OBSERVED_REPORT.read_text(encoding="utf-8")
        self.assertIn("# 02 Capstone Model Building 실행 관측", report)
        self.assertIn("## Problem statement / non-goals", report)
        self.assertIn("## Dataset / model / eval contract", report)
        self.assertIn("## Acceptance gates", report)
        self.assertIn("## Risk register", report)
        self.assertIn("## Failure-analysis outline", report)
        self.assertIn("Recall@10", report)
        self.assertNotIn(str(ROOT), report)

    def _run_json(self, relative_path: str) -> dict:
        completed = subprocess.run(
            [sys.executable, relative_path],
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
                f"{relative_path} did not emit JSON.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            ) from exc

    def _cleanup_generated_outputs(self) -> None:
        for directory in [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]:
            if directory.exists():
                shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
