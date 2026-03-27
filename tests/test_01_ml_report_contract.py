from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / '01_ml'
STAGES = [
    '01_tabular_classification',
    '02_tabular_regression',
    '03_model_selection_and_interpretation',
    '04_large_scale_tabular',
]


class TestMLStageLayout(unittest.TestCase):
    def test_stage_dirs_have_study_docs_and_local_code(self) -> None:
        self.assertTrue((ML_ROOT / 'README.md').exists())
        self.assertTrue((ML_ROOT / 'THEORY.md').exists())
        self.assertTrue((ML_ROOT / 'RESULTS.md').exists())
        self.assertTrue((ML_ROOT / 'run_all.py').exists())
        for stage in STAGES:
            stage_dir = ML_ROOT / stage
            self.assertTrue((stage_dir / 'README.md').exists(), f'missing stage README: {stage_dir}')
            self.assertTrue((stage_dir / 'THEORY.md').exists(), f'missing stage THEORY: {stage_dir}')
            self.assertTrue((stage_dir / 'run_stage.py').exists(), f'missing run_stage.py: {stage_dir}')
            self.assertTrue((stage_dir / 'experiment.py').exists(), f'missing experiment.py: {stage_dir}')
            self.assertTrue((stage_dir / 'dataset.py').exists(), f'missing dataset.py: {stage_dir}')
            self.assertTrue((stage_dir / 'artifacts').exists(), f'missing artifacts dir: {stage_dir}')

    def test_latest_artifact_contains_core_files(self) -> None:
        for stage in STAGES:
            artifact_root = ML_ROOT / stage / 'artifacts'
            run_dirs = sorted([p for p in artifact_root.iterdir() if p.is_dir()])
            self.assertTrue(run_dirs, f'no artifacts under {artifact_root}')
            latest = run_dirs[-1]
            self.assertTrue((latest / 'README.md').exists(), f'missing README: {latest}')
            self.assertTrue((latest / 'summary.md').exists(), f'missing summary: {latest}')
            self.assertTrue((latest / 'metrics.json').exists(), f'missing metrics: {latest}')
            self.assertTrue((latest / 'figures' / 'results').exists(), f'missing result figures: {latest}')
            self.assertTrue((latest / 'figures' / 'analysis').exists(), f'missing analysis figures: {latest}')
            self.assertGreaterEqual(len(list((latest / 'figures' / 'results').glob('*.svg'))), 3)
            self.assertGreaterEqual(len(list((latest / 'figures' / 'analysis').glob('*.svg'))), 3)
            readme_text = (latest / 'README.md').read_text(encoding='utf-8')
            summary_text = (latest / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('# ', readme_text)
            self.assertTrue('THEORY' in readme_text or 'THEORY.md' in readme_text or '이론' in readme_text)
            self.assertTrue('figures/results/' in readme_text or '.svg' in readme_text)
            self.assertIn('# ', summary_text)

    def test_metrics_have_primary_and_best_model(self) -> None:
        for metrics_path in ML_ROOT.glob('*/artifacts/*/metrics.json'):
            data = json.loads(metrics_path.read_text(encoding='utf-8'))
            self.assertIn('primary_metric', data)
            self.assertIn('best_model', data)
            self.assertIn('models', data)
            self.assertIn(data['best_model'], data['models'])
            self.assertTrue(data['models'], f'empty models in {metrics_path}')

    def test_results_index_links_local_artifacts(self) -> None:
        index_text = (ML_ROOT / 'RESULTS.md').read_text(encoding='utf-8')
        self.assertIn('01 ML 결과 인덱스', index_text)
        self.assertIn('artifacts/', index_text)
        self.assertIn('![](01_tabular_classification/artifacts/', index_text)
        self.assertIn('![](04_large_scale_tabular/artifacts/', index_text)


if __name__ == '__main__':
    unittest.main()
