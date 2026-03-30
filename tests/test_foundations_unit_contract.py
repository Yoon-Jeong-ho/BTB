from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONS = ROOT / '00_foundations'
UNIT = FOUNDATIONS / '01_tensor_shapes'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
FRAMEWORK_METRICS = ARTIFACTS / 'framework-manual' / 'metrics.json'
ANALYSIS_MD = UNIT / 'analysis.md'
GITKEEP = ARTIFACTS / '.gitkeep'
REQUIRED = [
    'README.md',
    'THEORY.md',
    'PREREQS.md',
    'lesson.yaml',
    'scratch_lab.py',
    'framework_lab.py',
    'analysis.py',
    'analysis.md',
    'reflection.md',
    'artifacts',
]


class TestFoundationsUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, relative_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _cleanup_generated_outputs(self) -> None:
        for path in (SCRATCH_METRICS, FRAMEWORK_METRICS, ANALYSIS_MD):
            if path.exists():
                path.unlink()

    def test_tensor_shapes_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_tensor_shapes_metadata_mentions_outputs(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('analysis_questions:', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        self.assertTrue(GITKEEP.exists(), 'artifacts/.gitkeep')
        self.assertEqual('', GITKEEP.read_text(encoding='utf-8'))

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self._cleanup_generated_outputs()

        scratch_result = self._run('00_foundations/01_tensor_shapes/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('00_foundations/01_tensor_shapes/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('00_foundations/01_tensor_shapes/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        analysis_text = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual([2, 4], scratch['matmul_shape'])
        self.assertEqual([2, 3], scratch['broadcast_result_shape'])
        self.assertIn('broadcast', scratch['mismatch_error'])
        self.assertEqual([4, 3], framework['logits_shape'])
        self.assertEqual([1.0, 1.0, 1.0, 1.0], framework['row_probability_sums'])
        self.assertIn('# 01 Tensor Shapes 분석', analysis_text)
        self.assertIn('## 해석', analysis_text)
        self.assertIn('shape mismatch', analysis_text)

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self._cleanup_generated_outputs()

        result = self._run('00_foundations/01_tensor_shapes/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = (result.stdout + result.stderr)
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)


if __name__ == '__main__':
    unittest.main()
