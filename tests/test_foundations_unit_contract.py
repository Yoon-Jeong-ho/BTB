from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONS = ROOT / '00_foundations'
TENSOR_UNIT = FOUNDATIONS / '01_tensor_shapes'
GPU_UNIT = FOUNDATIONS / '05_gpu_memory_runtime'
TENSOR_ARTIFACTS = TENSOR_UNIT / 'artifacts'
GPU_ARTIFACTS = GPU_UNIT / 'artifacts'
TENSOR_SCRATCH_METRICS = TENSOR_ARTIFACTS / 'scratch-manual' / 'metrics.json'
TENSOR_FRAMEWORK_METRICS = TENSOR_ARTIFACTS / 'framework-manual' / 'metrics.json'
TENSOR_ANALYSIS_MD = TENSOR_UNIT / 'analysis.md'
GPU_SCRATCH_METRICS = GPU_ARTIFACTS / 'scratch-manual' / 'metrics.json'
GPU_FRAMEWORK_METRICS = GPU_ARTIFACTS / 'framework-manual' / 'metrics.json'
GPU_OBSERVED_REPORT = GPU_ARTIFACTS / 'analysis-manual' / 'latest_report.md'
GPU_ANALYSIS_MD = GPU_UNIT / 'analysis.md'
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
GPU_GENERATED_DIRS = [
    GPU_ARTIFACTS / 'scratch-manual',
    GPU_ARTIFACTS / 'framework-manual',
    GPU_ARTIFACTS / 'analysis-manual',
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

    def _cleanup_generated_outputs(self, *paths: Path) -> None:
        for path in paths:
            if path.exists():
                path.unlink()

    def _cleanup_gpu_generated_artifacts(self) -> None:
        self._cleanup_generated_outputs(
            GPU_SCRATCH_METRICS,
            GPU_FRAMEWORK_METRICS,
            GPU_OBSERVED_REPORT,
        )
        for directory in GPU_GENERATED_DIRS:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    def test_tensor_shapes_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((TENSOR_UNIT / rel).exists(), rel)

    def test_tensor_shapes_metadata_mentions_outputs(self) -> None:
        text = (TENSOR_UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('analysis_questions:', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        for unit_artifacts in (TENSOR_ARTIFACTS, GPU_ARTIFACTS):
            gitkeep = unit_artifacts / '.gitkeep'
            self.assertTrue(gitkeep.exists(), f'{unit_artifacts}/.gitkeep')
            self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_gpu_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_gpu_generated_artifacts)
        self._cleanup_gpu_generated_artifacts()

        result = self._run('00_foundations/05_gpu_memory_runtime/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_gpu_memory_runtime_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((GPU_UNIT / rel).exists(), rel)

    def test_gpu_metadata_mentions_runtime_hooks(self) -> None:
        text = (GPU_UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('runtime_observation_hooks:', text)
        self.assertIn('max_memory_allocated', text)
        self.assertIn('analysis_questions:', text)

    def test_gpu_unit_labs_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_gpu_generated_artifacts)
        self._cleanup_gpu_generated_artifacts()

        scratch_result = self._run('00_foundations/05_gpu_memory_runtime/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('00_foundations/05_gpu_memory_runtime/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('00_foundations/05_gpu_memory_runtime/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(GPU_SCRATCH_METRICS.exists(), 'gpu scratch metrics missing')
        self.assertTrue(GPU_FRAMEWORK_METRICS.exists(), 'gpu framework metrics missing')
        self.assertTrue(GPU_OBSERVED_REPORT.exists(), 'gpu observed report missing')
        self.assertTrue(GPU_ANALYSIS_MD.exists(), 'gpu analysis.md missing')

        scratch = json.loads(GPU_SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(GPU_FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        observed_text = GPU_OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis_text = GPU_ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertGreater(scratch['batch_fp32_bytes'], scratch['batch_fp16_bytes'])
        self.assertEqual(2.0, scratch['dtype_savings_ratio'])
        self.assertIn(framework['device'], {'cpu', 'cuda'})
        self.assertGreaterEqual(framework['inference_runtime_ms'], 0.0)
        self.assertGreaterEqual(framework['training_runtime_ms'], 0.0)
        self.assertGreater(framework['training_grad_bytes'], 0)
        if framework['device'] == 'cuda':
            self.assertGreaterEqual(
                framework['training_max_memory_allocated'],
                framework['inference_max_memory_allocated'],
            )
            self.assertGreaterEqual(
                framework['training_max_memory_reserved'],
                framework['training_max_memory_allocated'],
            )
        else:
            self.assertEqual(0, framework['training_max_memory_allocated'])
            self.assertEqual(0, framework['training_max_memory_reserved'])
        self.assertIn('# 05 GPU Memory Runtime 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('training runtime', observed_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('반복 실행 시 불필요한 diff', analysis_text)
        self.assertNotIn(f'`{framework["training_runtime_ms"]} ms`', analysis_text)

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self._cleanup_generated_outputs(
            TENSOR_SCRATCH_METRICS,
            TENSOR_FRAMEWORK_METRICS,
            TENSOR_ANALYSIS_MD,
        )

        scratch_result = self._run('00_foundations/01_tensor_shapes/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('00_foundations/01_tensor_shapes/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('00_foundations/01_tensor_shapes/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(TENSOR_SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(TENSOR_FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(TENSOR_ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(TENSOR_SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(TENSOR_FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        analysis_text = TENSOR_ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual([2, 4], scratch['matmul_shape'])
        self.assertEqual([2, 3], scratch['broadcast_result_shape'])
        self.assertIn('broadcast', scratch['mismatch_error'])
        self.assertEqual([4, 3], framework['logits_shape'])
        self.assertEqual([1.0, 1.0, 1.0, 1.0], framework['row_probability_sums'])
        self.assertIn('# 01 Tensor Shapes 분석', analysis_text)
        self.assertIn('## 해석', analysis_text)
        self.assertIn('shape mismatch', analysis_text)

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self._cleanup_generated_outputs(
            TENSOR_SCRATCH_METRICS,
            TENSOR_FRAMEWORK_METRICS,
        )

        result = self._run('00_foundations/01_tensor_shapes/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)


if __name__ == '__main__':
    unittest.main()
