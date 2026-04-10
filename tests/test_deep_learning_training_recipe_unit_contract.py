from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '07_training_recipes_and_debugging'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'recipe_comparison.svg'
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


class TestDeepLearningTrainingRecipeUnitContract(unittest.TestCase):
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

    def test_readme_and_theory_are_korean_first_and_include_execution_examples(self) -> None:
        readme_text = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory_text = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertRegex(readme_text, r'[가-힣]')
        self.assertRegex(theory_text, r'[가-힣]')
        self.assertIn('실행 결과 예시', readme_text)
        self.assertIn('recipe_comparison.svg', readme_text)
        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('single-batch overfit', theory_text)

    def test_lesson_metadata_mentions_outputs_and_analysis_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('weight decay', text)
        self.assertIn('single-batch overfit', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/07_training_recipes_and_debugging/analysis.py')

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

        scratch_result = self._run('02_deep_learning/07_training_recipes_and_debugging/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/07_training_recipes_and_debugging/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/07_training_recipes_and_debugging/analysis.py')
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

        self.assertEqual(7, scratch['seed'])
        self.assertEqual('artifacts/scratch-manual/recipe_comparison.svg', scratch['figure_path'])
        self.assertIn('recipes', scratch)
        self.assertIn('sanity_checks', scratch)
        self.assertIn('small_batch_baseline', scratch['recipes'])
        self.assertIn('weight_decay_scheduler', scratch['recipes'])
        self.assertTrue(scratch['sanity_checks']['single_batch_overfit_passed'])
        self.assertTrue(scratch['sanity_checks']['high_lr_detected'])
        self.assertTrue(scratch['sanity_checks']['label_bug_detected'])
        self.assertGreater(
            scratch['recipes']['large_batch_constant_lr']['final_train_loss'],
            scratch['recipes']['small_batch_baseline']['final_train_loss'],
        )
        self.assertGreater(
            scratch['debug_probes']['shifted_label_bug']['final_val_loss'],
            scratch['recipes']['small_batch_baseline']['final_val_loss'],
        )
        self.assertIn('diverged', scratch['recipes']['high_lr_divergence']['alerts'])

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('tiny_mlp_gelu', framework['model_name'])
        self.assertIn('recipes', framework)
        self.assertIn('baseline_tiny_mlp', framework['recipes'])
        self.assertIn('weight_decay_scheduler_tiny_mlp', framework['recipes'])
        self.assertTrue(framework['sanity_checks']['single_batch_overfit_passed'])
        self.assertTrue(framework['sanity_checks']['high_lr_detected'])
        self.assertTrue(framework['sanity_checks']['label_bug_detected'])
        self.assertLess(
            framework['recipes']['weight_decay_scheduler_tiny_mlp']['final_val_loss'],
            framework['recipes']['baseline_tiny_mlp']['final_val_loss'],
        )
        self.assertIn('diverged', framework['recipes']['high_lr_tiny_mlp']['alerts'])

        self.assertIn('<svg', figure_text)
        self.assertIn('Training recipe comparison (scratch)', figure_text)
        self.assertIn('# 07 학습 레시피와 디버깅 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](../../THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
