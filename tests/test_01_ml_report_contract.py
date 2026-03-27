from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = ROOT / "reports" / "01_ml"
STAGES = [
    "01_tabular_classification",
    "02_tabular_regression",
    "03_model_selection_and_interpretation",
    "04_large_scale_tabular",
]


class TestMLReportContract(unittest.TestCase):
    def test_stage_reports_exist_with_core_artifacts(self) -> None:
        self.assertTrue(REPORTS_ROOT.exists(), f"missing reports root: {REPORTS_ROOT}")
        for stage in STAGES:
            stage_dir = REPORTS_ROOT / stage
            theory_path = ROOT / "01_ml" / stage / "THEORY.md"
            self.assertTrue(theory_path.exists(), f"missing theory doc: {theory_path}")
            self.assertTrue(stage_dir.exists(), f"missing stage dir: {stage_dir}")
            run_dirs = sorted([p for p in stage_dir.iterdir() if p.is_dir()])
            self.assertTrue(run_dirs, f"no report run dir under {stage_dir}")
            latest = run_dirs[-1]
            self.assertTrue((latest / "summary.md").exists(), f"missing summary: {latest}")
            self.assertTrue((latest / "metrics.json").exists(), f"missing metrics: {latest}")
            self.assertTrue((latest / "figures" / "results").exists(), f"missing result figures: {latest}")
            self.assertTrue((latest / "figures" / "analysis").exists(), f"missing analysis figures: {latest}")
            self.assertGreaterEqual(len(list((latest / "figures" / "results").glob("*.svg"))), 3)
            self.assertGreaterEqual(len(list((latest / "figures" / "analysis").glob("*.svg"))), 3)
            summary_text = (latest / "summary.md").read_text(encoding="utf-8")
            self.assertIn("# ", summary_text)
            self.assertIn("THEORY.md", summary_text)
            self.assertTrue("figures/results/" in summary_text or ".svg" in summary_text)
            self.assertTrue("figures/analysis/" in summary_text or ".svg" in summary_text)

    def test_metrics_have_primary_and_best_model(self) -> None:
        for metrics_path in REPORTS_ROOT.glob("*/*/metrics.json"):
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("primary_metric", data)
            self.assertIn("best_model", data)
            self.assertIn("models", data)
            self.assertIn(data["best_model"], data["models"])
            self.assertTrue(data["models"], f"empty models in {metrics_path}")

    def test_track_index_is_korean_and_links_previews(self) -> None:
        index_text = (REPORTS_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("01 ML 리포트 인덱스", index_text)
        self.assertIn("![](01_tabular_classification/", index_text)
        self.assertIn("![](04_large_scale_tabular/", index_text)


if __name__ == "__main__":
    unittest.main()
