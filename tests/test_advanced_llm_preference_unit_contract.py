from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '05_preference_optimization_dpo_orpo_kto'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'preference_margin.svg'
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


class TestAdvancedLLMPreferenceUnitContract(unittest.TestCase):
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
        self.assertIn('python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py', readme)
        self.assertIn('preference_margin.svg', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)

        for text in (readme, theory, reflection):
            self.assertIn('chosen', text)
            self.assertIn('rejected', text)
            self.assertIn('log-prob margin', text)
            self.assertIn('DPO', text)
            self.assertIn('ORPO', text)
            self.assertIn('KTO', text)
            self.assertIn('full RL', text)
            self.assertIn('alignment', text)
            self.assertIn('eval', text)

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
        self.assertIn('chosen/rejected pair', text)
        self.assertIn('log-prob margin', text)
        self.assertIn('policy update without full RL', text)
        self.assertIn('alignment eval tradeoff', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py')
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

        self.assertEqual(4, scratch['preference_batch']['prompt_count'])
        self.assertEqual(4, scratch['preference_batch']['pair_count'])
        self.assertGreater(scratch['preference_batch']['desirable_labels'], 0)
        self.assertGreater(scratch['preference_batch']['undesirable_labels'], 0)
        self.assertEqual(set(scratch['objective_views']), {'dpo', 'orpo', 'kto'})
        self.assertTrue(scratch['objective_views']['dpo']['requires_chosen_rejected_pairs'])
        self.assertTrue(scratch['objective_views']['orpo']['requires_chosen_rejected_pairs'])
        self.assertFalse(scratch['objective_views']['kto']['requires_chosen_rejected_pairs'])
        self.assertIn('reference-relative chosen/rejected log-prob margin', scratch['objective_views']['dpo']['signal'])
        self.assertGreater(scratch['margin_summary']['avg_policy_margin'], 0.0)
        self.assertGreater(scratch['margin_summary']['avg_dpo_advantage'], 0.0)
        self.assertEqual('style_over_factuality', scratch['alignment_eval']['primary_tradeoff_watch'])
        self.assertTrue(scratch['alignment_eval']['length_bias_flag'])
        self.assertIn('artifacts/scratch-manual/preference_margin.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Preference optimization margins', figure_text)
        self.assertIn('chosen-rejected margin', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('tiny_numeric_policy', framework['simulation'])
        self.assertEqual(set(framework['objective_losses']), {'dpo', 'orpo', 'kto'})
        self.assertGreater(framework['policy_update']['avg_margin_after'], framework['policy_update']['avg_margin_before'])
        self.assertGreater(framework['policy_update']['pair_accuracy_after'], framework['policy_update']['pair_accuracy_before'])
        self.assertLessEqual(
            framework['policy_update']['reference_drift_after'],
            framework['policy_update']['reference_drift_guardrail'],
        )
        self.assertTrue(framework['policy_update']['without_full_rl_loop'])
        self.assertEqual('DPO', framework['contrast']['pairwise_reference_method'])
        self.assertEqual('KTO', framework['contrast']['label_only_method'])
        self.assertGreater(
            framework['eval_tradeoffs']['helpfulness_gain_proxy'],
            framework['eval_tradeoffs']['refusal_overreach_delta'],
        )

        self.assertIn('# 05 Preference Optimization 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('DPO', observed_text)
        self.assertIn('ORPO', observed_text)
        self.assertIn('KTO', observed_text)
        self.assertIn('chosen/rejected', observed_text)
        self.assertIn('log-prob margin', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
