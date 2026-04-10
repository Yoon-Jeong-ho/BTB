from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '02_cnn_and_image_classification'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'cnn_feature_maps.svg'
FRAMEWORK_METRICS = ARTIFACTS / 'framework-manual' / 'metrics.json'
OBSERVED_REPORT = ARTIFACTS / 'analysis-manual' / 'latest_report.md'
ANALYSIS_MD = UNIT / 'analysis.md'
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
GENERATED_DIRS = [
    ARTIFACTS / 'scratch-manual',
    ARTIFACTS / 'framework-manual',
    ARTIFACTS / 'analysis-manual',
]


class TestDeepLearningCnnUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, relative_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
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

    def _cleanup_generated_outputs(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            if path.exists():
                path.unlink()
        for directory in GENERATED_DIRS:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    def _korean_first(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:8]:
            if line.startswith('```'):
                continue
            if re.search(r'[가-힣]', line):
                return True
        return False

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_readme_and_theory_are_korean_first_with_execution_examples(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertTrue(self._korean_first(readme))
        self.assertTrue(self._korean_first(theory))
        self.assertIn('> Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('python 02_deep_learning/02_cnn_and_image_classification/scratch_lab.py', readme)
        self.assertIn('python 02_deep_learning/02_cnn_and_image_classification/framework_lab.py', readme)
        self.assertIn('cnn_feature_maps.svg', readme)

        self.assertIn('실행 결과 예시', theory)
        self.assertIn('local receptive field', theory)
        self.assertIn('pooling', theory)
        self.assertIn('feature map', theory)
        self.assertIn('image classification', theory)

    def test_lesson_metadata_mentions_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch metrics json', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('framework metrics json', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('observed analysis report', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('local receptive field', text)
        self.assertIn('pooling', text)
        self.assertIn('feature map', text)
        self.assertIn('image classification baseline', text)

    def test_artifacts_gitkeep_is_empty(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/02_cnn_and_image_classification/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_scratch_framework_and_analysis_generate_expected_outputs(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('02_deep_learning/02_cnn_and_image_classification/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/02_cnn_and_image_classification/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/02_cnn_and_image_classification/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'observed report missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure_text = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed_text = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis_text = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual([4, 3, 6, 6], scratch['dataset_shape'])
        self.assertEqual(3, scratch['input_channel_count'])
        self.assertEqual(2, scratch['output_feature_map_count'])
        self.assertEqual([3, 3], scratch['local_receptive_field'])
        self.assertEqual([2, 3, 3, 3], scratch['conv_kernel_shape'])
        self.assertEqual([4, 2, 4, 4], scratch['feature_map_shape'])
        self.assertEqual([4, 2, 2, 2], scratch['pooled_shape'])
        self.assertEqual(16, scratch['parameter_sharing_reuse_count'])
        self.assertGreater(scratch['pooling_reduction_ratio'], 1.0)
        self.assertGreaterEqual(scratch['classification_accuracy'], 0.99)
        self.assertEqual('artifacts/scratch-manual/cnn_feature_maps.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('CNN feature maps', figure_text)

        self.assertIn(framework['backend'], {'pytorch', 'python-fallback'})
        self.assertEqual('cpu', framework['device'])
        self.assertEqual(3, framework['input_channel_count'])
        self.assertGreaterEqual(framework['output_feature_map_count'], 2)
        self.assertEqual(framework['dataset_shape'][1], framework['input_channel_count'])
        self.assertEqual(framework['conv_weight_shape'][0], framework['output_feature_map_count'])
        self.assertEqual(framework['conv_weight_shape'][1], framework['input_channel_count'])
        self.assertEqual(framework['feature_map_shape'][0], framework['dataset_shape'][0])
        self.assertEqual(framework['pooled_shape'][0], framework['dataset_shape'][0])
        self.assertEqual(framework['logits_shape'][0], framework['dataset_shape'][0])
        self.assertEqual(framework['logits_shape'][1], len(framework['class_names']))
        self.assertEqual(len(framework['predictions']), framework['dataset_shape'][0])
        self.assertGreaterEqual(framework['accuracy'], 0.99)

        self.assertIn('# 02 CNN and Image Classification 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)
        self.assertNotIn(str(framework['accuracy']), analysis_text)

        OBSERVED_REPORT.write_text('오래된 관측', encoding='utf-8')
        rerun_result = self._run('02_deep_learning/02_cnn_and_image_classification/analysis.py')
        self.assertEqual(0, rerun_result.returncode, rerun_result.stderr)
        self.assertEqual(stable_before, ANALYSIS_MD.read_text(encoding='utf-8'))
        observed_after = OBSERVED_REPORT.read_text(encoding='utf-8')
        self.assertNotEqual('오래된 관측', observed_after)
        self.assertIn('# 02 CNN and Image Classification 실행 관측', observed_after)


if __name__ == '__main__':
    unittest.main()
