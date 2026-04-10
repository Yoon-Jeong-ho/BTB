from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '01_language_modeling_and_pretraining_objectives'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'objective_comparison.svg'
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


class TestAdvancedLLMObjectivesUnitContract(unittest.TestCase):
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

    def test_docs_are_korean_first_and_show_real_execution_examples(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        prereqs = (UNIT / 'PREREQS.md').read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')

        self.assertRegex('\n'.join(readme.splitlines()[:8]), r'[가-힣]')
        self.assertRegex('\n'.join(theory.splitlines()[:8]), r'[가-힣]')
        self.assertRegex('\n'.join(prereqs.splitlines()[:8]), r'[가-힣]')
        self.assertRegex('\n'.join(reflection.splitlines()[:8]), r'[가-힣]')

        self.assertIn('> Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py', readme)
        self.assertIn('causal LM', readme)
        self.assertIn('masked LM', readme)
        self.assertIn('span corruption', readme)
        self.assertIn('loss-mask density', readme)
        self.assertIn('context window', readme)
        self.assertIn('objective_comparison.svg', readme)

        self.assertIn('실행 결과 예시', theory)
        self.assertIn('target framing', theory)
        self.assertIn('loss-mask density', theory)
        self.assertIn('context window', theory)
        self.assertIn('causal LM', theory)
        self.assertIn('masked LM', theory)
        self.assertIn('span corruption', theory)

        self.assertIn('학습자', reflection)
        self.assertIn('causal LM', reflection)
        self.assertIn('masked LM', reflection)
        self.assertIn('span corruption', reflection)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch metrics json', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('framework metrics json', text)
        self.assertIn('observed analysis report', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('reflection markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('causal LM', text)
        self.assertIn('masked LM', text)
        self.assertIn('span corruption', text)
        self.assertIn('target framing', text)
        self.assertIn('context window', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives/analysis.py')
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

        self.assertEqual(set(scratch['objectives']), {'causal_lm', 'masked_lm', 'span_corruption'})
        self.assertGreater(
            scratch['objectives']['causal_lm']['loss_mask_density'],
            scratch['objectives']['masked_lm']['loss_mask_density'],
        )
        self.assertGreater(
            scratch['objectives']['span_corruption']['loss_mask_density'],
            scratch['objectives']['masked_lm']['loss_mask_density'],
        )
        self.assertEqual('causal_lm', scratch['comparisons']['densest_supervision'])
        self.assertEqual('masked_lm', scratch['comparisons']['sparsest_supervision'])
        self.assertTrue(scratch['comparisons']['causal_future_blocked'])
        self.assertTrue(scratch['comparisons']['masked_middle_token_sees_both_sides'])
        self.assertTrue(scratch['comparisons']['span_decoder_reads_previous_targets_only'])
        self.assertIn('artifacts/scratch-manual/objective_comparison.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Pretraining objective comparison', figure_text)
        self.assertIn('Loss mask density', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(11, framework['vocab_size'])
        self.assertEqual(set(framework['objectives']), {'causal_lm', 'masked_lm', 'span_corruption'})
        self.assertEqual('causal_lm', framework['density_ranking'][0])
        self.assertEqual('masked_lm', framework['density_ranking'][-1])
        self.assertGreater(framework['objectives']['causal_lm']['scored_tokens'], framework['objectives']['masked_lm']['scored_tokens'])
        self.assertGreater(framework['objectives']['span_corruption']['decoder_target_length'], 0)
        self.assertTrue(framework['context_window']['causal_future_blocked'])
        self.assertTrue(framework['context_window']['masked_middle_token_sees_both_sides'])
        self.assertTrue(framework['context_window']['span_decoder_reads_previous_targets_only'])
        for objective in framework['objectives'].values():
            self.assertGreater(objective['mean_loss'], 0.0)

        self.assertIn('# 01 Language Modeling and Pretraining Objectives 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('causal LM', observed_text)
        self.assertIn('masked LM', observed_text)
        self.assertIn('span corruption', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
