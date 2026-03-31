from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '03_nlp' / '01_text_classification'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_DIR / 'token_signal.svg'
FRAMEWORK_METRICS = FRAMEWORK_DIR / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_DIR / 'latest_report.md'
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

GENERATED_DIRS = [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]


class TestNlpTaskUnitContract(unittest.TestCase):
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
        for directory in GENERATED_DIRS:
            if directory.exists():
                shutil.rmtree(directory)

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_readme_and_theory_are_korean_first_and_include_examples(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertRegex(readme, r'[가-힣]')
        self.assertRegex(theory, r'[가-힣]')
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('metrics.json', readme)
        self.assertIn('실행 결과 예시', theory)
        self.assertIn('bag-of-words', theory)
        self.assertIn('PyTorch', theory)

    def test_lesson_metadata_mentions_required_outputs(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', lesson)
        self.assertIn('scratch svg figure', lesson)
        self.assertIn('analysis_questions:', lesson)
        self.assertIn('macro F1', lesson)
        self.assertIn('bag-of-words', lesson)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('03_nlp/01_text_classification/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        scratch_result = self._run('03_nlp/01_text_classification/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('03_nlp/01_text_classification/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('03_nlp/01_text_classification/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch svg figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'analysis observed report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual('artifacts/scratch-manual/token_signal.svg', scratch['figure_path'])
        self.assertGreaterEqual(scratch['train_size'], 6)
        self.assertGreaterEqual(scratch['eval_accuracy'], 0.5)
        self.assertGreaterEqual(scratch['eval_macro_f1'], 0.5)
        self.assertIn('class_priors', scratch)
        self.assertIn('top_positive_tokens', scratch)
        self.assertIn('top_negative_tokens', scratch)
        self.assertIn('<svg', SCRATCH_FIGURE.read_text(encoding='utf-8'))

        self.assertGreaterEqual(framework['train_size'], 6)
        self.assertGreaterEqual(framework['eval_accuracy'], 0.5)
        self.assertGreaterEqual(framework['eval_macro_f1'], 0.5)
        self.assertEqual(2, framework['num_classes'])
        self.assertGreater(framework['vocab_size'], 5)
        self.assertEqual(2, len(framework['label_names']))
        self.assertIn('loss_history_head', framework)
        self.assertIn('prediction_rows', framework)

        self.assertIn('# 01 Text Classification 실행 관측', observed)
        self.assertIn('## 한국어 해석', observed)
        self.assertIn('[THEORY.md](./THEORY.md)', observed)
        self.assertIn('latest_report.md', analysis)
        self.assertIn('## 관련 이론', analysis)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis)
        self.assertNotIn(str(framework['eval_accuracy']), analysis)


if __name__ == '__main__':
    unittest.main()
