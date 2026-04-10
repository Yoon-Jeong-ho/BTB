from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '03_sequence_models_rnn_lstm_gru'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_DIR / 'hidden_state_diagnostics.svg'
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


class TestDeepLearningSequenceUnitContract(unittest.TestCase):
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
        self.assertIn('Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('hidden_state_diagnostics.svg', readme)
        self.assertIn('실행 결과 예시', theory)
        self.assertIn('teacher forcing', theory)
        self.assertIn('LSTM', theory)
        self.assertIn('GRU', theory)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', lesson)
        self.assertIn('required_outputs:', lesson)
        self.assertIn('scratch svg figure', lesson)
        self.assertIn('framework metrics json', lesson)
        self.assertIn('analysis_questions:', lesson)
        self.assertIn('teacher forcing', lesson)
        self.assertIn('hidden state', lesson)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/03_sequence_models_rnn_lstm_gru/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    @unittest.skipIf(torch is None, 'PyTorch not installed; skipping framework run contract')
    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('02_deep_learning/03_sequence_models_rnn_lstm_gru/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/03_sequence_models_rnn_lstm_gru/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/03_sequence_models_rnn_lstm_gru/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch SVG missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'analysis observed report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis = ANALYSIS_MD.read_text(encoding='utf-8')

        for key in (
            'rnn_order_cosine_gap',
            'lstm_order_cosine_gap',
            'gru_order_cosine_gap',
            'rnn_long_range_signal',
            'lstm_long_range_signal',
            'gru_long_range_signal',
            'teacher_forcing_loss',
            'free_running_loss',
            'teacher_forcing_gap',
            'figure_path',
        ):
            self.assertIn(key, scratch)

        self.assertGreater(scratch['rnn_order_cosine_gap'], 0.0)
        self.assertGreater(scratch['lstm_order_cosine_gap'], 0.0)
        self.assertGreater(scratch['gru_order_cosine_gap'], 0.0)
        self.assertGreater(scratch['lstm_long_range_signal'], scratch['rnn_long_range_signal'])
        self.assertGreater(scratch['gru_long_range_signal'], scratch['rnn_long_range_signal'])
        self.assertGreater(scratch['free_running_loss'], scratch['teacher_forcing_loss'])
        self.assertGreater(scratch['teacher_forcing_gap'], 0.0)
        self.assertEqual('artifacts/scratch-manual/hidden_state_diagnostics.svg', scratch['figure_path'])
        self.assertIn('<svg', figure)
        self.assertIn('Hidden state diagnostics', figure)

        for key in (
            'device',
            'hidden_shapes',
            'rnn_order_cosine_gap',
            'lstm_order_cosine_gap',
            'gru_order_cosine_gap',
            'rnn_long_range_signal',
            'lstm_long_range_signal',
            'gru_long_range_signal',
            'teacher_forcing_loss',
            'free_running_loss',
            'teacher_forcing_gap',
            'decoder_logits_shape',
        ):
            self.assertIn(key, framework)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(['gru', 'lstm_c', 'lstm_h', 'rnn'], sorted(framework['hidden_shapes'].keys()))
        self.assertGreater(framework['rnn_order_cosine_gap'], 0.0)
        self.assertGreater(framework['lstm_order_cosine_gap'], 0.0)
        self.assertGreater(framework['gru_order_cosine_gap'], 0.0)
        self.assertGreater(framework['lstm_long_range_signal'], framework['rnn_long_range_signal'])
        self.assertGreater(framework['gru_long_range_signal'], framework['rnn_long_range_signal'])
        self.assertGreater(framework['free_running_loss'], framework['teacher_forcing_loss'])
        self.assertGreater(framework['teacher_forcing_gap'], 0.0)
        self.assertEqual([2, 4, 6], framework['decoder_logits_shape'])

        self.assertIn('# 03 Sequence Models 실행 관측', observed)
        self.assertIn('## 한국어 해석', observed)
        self.assertIn('[THEORY.md](../../THEORY.md)', observed)
        self.assertIn('teacher forcing gap', observed)
        self.assertEqual(stable_before, analysis)
        self.assertIn('latest_report.md', analysis)
        self.assertIn('## 관련 이론', analysis)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis)


if __name__ == '__main__':
    unittest.main()
