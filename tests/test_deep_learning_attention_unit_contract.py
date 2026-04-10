from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '04_attention_and_transformers'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'attention_patterns.svg'
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


class TestDeepLearningAttentionUnitContract(unittest.TestCase):
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

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_readme_and_theory_are_korean_first_with_execution_examples(self) -> None:
        readme_text = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory_text = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertRegex('\n'.join(readme_text.splitlines()[:8]), r'[가-힣]')
        self.assertRegex('\n'.join(theory_text.splitlines()[:8]), r'[가-힣]')

        self.assertIn('Status: runnable', readme_text)
        self.assertIn('실행 결과 예시', readme_text)
        self.assertIn('scratch_lab.py', readme_text)
        self.assertIn('framework_lab.py', readme_text)
        self.assertIn('attention_patterns.svg', readme_text)

        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('multi-head', theory_text)
        self.assertIn('encoder', theory_text)
        self.assertIn('decoder', theory_text)
        self.assertIn('recurrent bottleneck', theory_text)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('multi-head', text)
        self.assertIn('encoder', text)
        self.assertIn('decoder', text)
        self.assertIn('recurrent bottleneck', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/04_attention_and_transformers/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('02_deep_learning/04_attention_and_transformers/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/04_attention_and_transformers/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/04_attention_and_transformers/analysis.py')
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

        self.assertEqual(scratch['sequence_length'], len(scratch['tokens']))
        self.assertEqual(2, scratch['multi_head']['head_count'])
        self.assertLess(scratch['max_row_sum_error'], 1e-6)
        self.assertTrue(scratch['encoder_decoder']['causal_mask_future_blocked'])
        self.assertLess(
            scratch['recurrent_relief']['attention_parallel_rounds'],
            scratch['recurrent_relief']['recurrent_steps'],
        )
        self.assertIn('artifacts/scratch-manual/attention_patterns.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Attention pattern heatmap', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(2, framework['num_heads'])
        self.assertTrue(framework['cross_attention_used'])
        self.assertLess(framework['decoder_future_attention_max'], 1e-6)
        self.assertGreater(framework['encoder_future_attention_mean'], 0.0)
        self.assertGreater(framework['per_head_difference_mean'], 0.0)
        self.assertEqual(framework['encoder_hidden_shape'], framework['decoder_hidden_shape'])
        self.assertEqual(framework['encoder_hidden_shape'], framework['decoder_block_output_shape'])
        self.assertLess(
            framework['recurrent_relief']['attention_parallel_rounds'],
            framework['recurrent_relief']['recurrent_steps'],
        )

        self.assertIn('# 04 Attention and Transformers 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
