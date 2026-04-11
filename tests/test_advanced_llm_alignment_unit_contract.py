from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '08_alignment_safety_and_model_behavior'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'alignment_behavior_slices.svg'
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


class TestAdvancedLLMAlignmentUnitContract(unittest.TestCase):
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
        self.assertIn('python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/analysis.py', readme)
        self.assertIn('alignment_behavior_slices.svg', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)

        for text in (readme, theory, reflection):
            self.assertIn('alignment vs capability', text)
            self.assertIn('refusal', text)
            self.assertIn('over-refusal', text)
            self.assertIn('harmlessness', text)
            self.assertIn('robustness', text)
            self.assertIn('behavioral eval', text)
            self.assertIn('slice analysis', text)
            self.assertIn('policy vs system-level safety', text)
            self.assertIn('model policy', text)
            self.assertIn('system guardrail', text)

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
        self.assertIn('alignment vs capability', text)
        self.assertIn('refusal vs over-refusal', text)
        self.assertIn('harmlessness and robustness', text)
        self.assertIn('behavioral eval slice analysis', text)
        self.assertIn('policy vs system-level safety', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/analysis.py')
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

        self.assertEqual('08_alignment_safety_and_model_behavior', scratch['setup']['unit'])
        self.assertTrue(scratch['setup']['cpu_safe'])
        self.assertEqual('toy_behavior_policy_eval', scratch['setup']['mode'])
        self.assertGreaterEqual(scratch['alignment_vs_capability']['capability_score'], 0.8)
        self.assertLess(
            scratch['alignment_vs_capability']['capability_score'],
            scratch['alignment_vs_capability']['behavior_contract_score'] + 0.15,
        )
        self.assertTrue(scratch['alignment_vs_capability']['capability_can_enable_risk'])
        self.assertEqual(6, scratch['behavior_slices']['prompt_count'])
        self.assertEqual(set(scratch['behavior_slices']['slice_names']), {'benign', 'harmful', 'borderline'})
        self.assertGreaterEqual(scratch['behavior_slices']['harmful_refusal_rate'], 0.9)
        self.assertGreaterEqual(scratch['behavior_slices']['benign_answer_rate'], 0.75)
        self.assertLessEqual(scratch['behavior_slices']['over_refusal_rate'], 0.25)
        self.assertGreaterEqual(scratch['behavior_slices']['safe_alternative_rate'], 0.5)
        self.assertEqual(0, scratch['refusal_confusion_matrix']['unsafe_compliance'])
        self.assertGreater(scratch['refusal_confusion_matrix']['harmful_refused'], 0)
        self.assertGreater(scratch['refusal_confusion_matrix']['borderline_safe_alternative'], 0)
        self.assertGreaterEqual(scratch['robustness_probe']['min_stability'], 0.75)
        self.assertFalse(scratch['robustness_probe']['jailbreak_variant_bypassed'])
        self.assertEqual('slice-based, not one scalar', scratch['behavioral_eval']['scoring_note'])
        self.assertTrue(scratch['policy_vs_system_level_safety']['requires_system_guardrails'])
        self.assertIn('tool permission gating', scratch['policy_vs_system_level_safety']['system_guardrails'])
        self.assertIn('unsafe content refusal', scratch['policy_vs_system_level_safety']['model_policy'])
        self.assertIn('artifacts/scratch-manual/alignment_behavior_slices.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Alignment behavior slices', figure_text)
        self.assertIn('over-refusal', figure_text)
        self.assertIn('robustness', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('deterministic_behavior_eval_simulation', framework['simulation'])
        self.assertEqual(8, framework['dataset_size'])
        self.assertEqual(['benign', 'harmful', 'borderline', 'robustness'], framework['slices'])
        self.assertEqual({'capability_only_assistant', 'aligned_assistant'}, set(framework['policy_candidates']))
        self.assertGreater(
            framework['aggregate_scores']['aligned_assistant']['behavior_contract_score'],
            framework['aggregate_scores']['capability_only_assistant']['behavior_contract_score'],
        )
        self.assertGreaterEqual(framework['slice_analysis']['harmful']['refusal_rate'], 0.9)
        self.assertGreaterEqual(framework['slice_analysis']['benign']['answer_rate'], 0.75)
        self.assertLessEqual(framework['slice_analysis']['benign']['over_refusal_rate'], 0.25)
        self.assertGreaterEqual(framework['slice_analysis']['borderline']['safe_alternative_rate'], 0.5)
        self.assertGreaterEqual(framework['slice_analysis']['robustness']['pass_rate'], 0.75)
        self.assertTrue(framework['behavior_eval']['slice_based'])
        self.assertTrue(framework['behavior_eval']['single_scalar_is_insufficient'])
        self.assertEqual('tool_permission_bypass', framework['policy_vs_system_level']['missing_guardrail_failure'])
        self.assertIn('moderation and audit logging', framework['policy_vs_system_level']['system_guardrails'])
        self.assertIn('refuse and redirect harmful requests', framework['policy_vs_system_level']['model_policy'])

        self.assertIn('# 08 Alignment, Safety, and Model Behavior 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('alignment vs capability', observed_text)
        self.assertIn('refusal', observed_text)
        self.assertIn('over-refusal', observed_text)
        self.assertIn('harmlessness', observed_text)
        self.assertIn('robustness', observed_text)
        self.assertIn('behavioral eval', observed_text)
        self.assertIn('policy vs system-level safety', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
