from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '01_perceptron_and_mlp'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_ARTIFACTS = ARTIFACTS / 'scratch-manual'
FRAMEWORK_ARTIFACTS = ARTIFACTS / 'framework-manual'
ANALYSIS_ARTIFACTS = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_ARTIFACTS / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_ARTIFACTS / 'decision_regions.svg'
FRAMEWORK_METRICS = FRAMEWORK_ARTIFACTS / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_ARTIFACTS / 'latest_report.md'
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
    SCRATCH_ARTIFACTS,
    FRAMEWORK_ARTIFACTS,
    ANALYSIS_ARTIFACTS,
]


class TestDeepLearningPerceptronUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, relative_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _cleanup_generated_artifacts(self) -> None:
        for directory in GENERATED_DIRS:
            if directory.exists():
                shutil.rmtree(directory)

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
        self.assertIn('실행 결과 예시', theory)
        self.assertIn('python 02_deep_learning/01_perceptron_and_mlp/scratch_lab.py', readme)
        self.assertIn('python 02_deep_learning/01_perceptron_and_mlp/framework_lab.py', readme)
        self.assertIn('decision_regions.svg', readme)
        self.assertIn('tiny MLP', theory)

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
        self.assertIn('XOR', text)
        self.assertIn('MLP', text)

    def test_artifacts_gitkeep_is_empty(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_artifacts)
        self._cleanup_generated_artifacts()

        result = self._run('02_deep_learning/01_perceptron_and_mlp/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_scratch_framework_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_generated_artifacts)
        self._cleanup_generated_artifacts()

        scratch_result = self._run('02_deep_learning/01_perceptron_and_mlp/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/01_perceptron_and_mlp/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/01_perceptron_and_mlp/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'observed report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure_text = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed_text = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis_text = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual('predict=1 if w·x + b >= 0 else 0', scratch['decision_rule'])
        self.assertEqual(1.0, scratch['linear_dataset_accuracy'])
        self.assertEqual(0.75, scratch['xor_best_accuracy'])
        self.assertTrue(scratch['linear_is_separable'])
        self.assertFalse(scratch['xor_is_separable_with_single_neuron'])
        self.assertEqual('artifacts/scratch-manual/decision_regions.svg', scratch['figure_path'])

        self.assertIn(framework['backend'], {'pytorch', 'python-fallback'})
        self.assertEqual('cpu', framework['device'])
        self.assertEqual(1.0, framework['single_neuron_linear_accuracy'])
        self.assertLessEqual(framework['single_neuron_xor_accuracy'], 0.75)
        self.assertGreaterEqual(framework['tiny_mlp_xor_accuracy'], 0.99)
        self.assertGreater(framework['xor_accuracy_gain'], 0.2)
        self.assertGreater(framework['tiny_mlp_parameter_count'], framework['single_neuron_parameter_count'])

        self.assertIn('<svg', figure_text)
        self.assertIn('Decision boundaries: perceptron vs XOR', figure_text)
        self.assertIn('# 01 Perceptron and MLP 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('XOR', observed_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('반복 실행 시 불필요한 diff', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)
        self.assertNotIn(str(framework['tiny_mlp_xor_accuracy']), analysis_text)

        stable_before = analysis_text
        OBSERVED_REPORT.write_text('오래된 관측', encoding='utf-8')
        rerun_result = self._run('02_deep_learning/01_perceptron_and_mlp/analysis.py')
        self.assertEqual(0, rerun_result.returncode, rerun_result.stderr)
        stable_after = ANALYSIS_MD.read_text(encoding='utf-8')
        observed_after = OBSERVED_REPORT.read_text(encoding='utf-8')
        self.assertEqual(stable_before, stable_after)
        self.assertNotEqual('오래된 관측', observed_after)
        self.assertIn('# 01 Perceptron and MLP 실행 관측', observed_after)


if __name__ == '__main__':
    unittest.main()
