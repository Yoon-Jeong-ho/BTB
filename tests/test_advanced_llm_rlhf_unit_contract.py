from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '06_rlhf_and_reasoning_rl'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'rlhf_reasoning_reward.svg'
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


class TestAdvancedLLMRLHFUnitContract(unittest.TestCase):
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
        self.assertIn('python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/analysis.py', readme)
        self.assertIn('rlhf_reasoning_reward.svg', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)

        for text in (readme, theory, reflection):
            self.assertIn('reward model', text)
            self.assertIn('RLHF', text)
            self.assertIn('PPO', text)
            self.assertIn('policy update', text)
            self.assertIn('verifier', text)
            self.assertIn('judge', text)
            self.assertIn('reasoning RL', text)
            self.assertIn('reward shaping', text)
            self.assertIn('reward hacking', text)
            self.assertIn('verbosity', text)
            self.assertIn('over-refusal', text)

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
        self.assertIn('reward model intuition', text)
        self.assertIn('PPO/RLHF high-level loop', text)
        self.assertIn('verifier/judge signal', text)
        self.assertIn('reasoning-oriented reward shaping', text)
        self.assertIn('reward hacking', text)
        self.assertIn('over-refusal', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/analysis.py')
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

        self.assertEqual('06_rlhf_and_reasoning_rl', scratch['setup']['unit'])
        self.assertTrue(scratch['setup']['cpu_safe'])
        self.assertEqual(4, scratch['reward_model_batch']['prompt_count'])
        self.assertEqual(8, scratch['reward_model_batch']['candidate_count'])
        self.assertEqual(4, scratch['reward_model_batch']['chosen_rejected_pairs'])
        self.assertGreater(scratch['reward_model_batch']['avg_reward_chosen'], scratch['reward_model_batch']['avg_reward_rejected'])
        self.assertEqual('preference proxy, not truth engine', scratch['reward_model_batch']['reward_model_intuition'])
        self.assertEqual(['sample_prompts', 'policy_rollouts', 'score_rewards', 'ppo_family_update', 'regression_eval'], scratch['rlhf_loop_view']['steps'])
        self.assertTrue(scratch['rlhf_loop_view']['kl_anchor_enabled'])
        self.assertIn('PPO-family', scratch['rlhf_loop_view']['policy_update_style'])
        self.assertGreater(scratch['reasoning_signal']['process_reward_weight'], 0.0)
        self.assertGreater(scratch['reasoning_signal']['verifier_pass_rate'], 0.0)
        self.assertGreater(scratch['reasoning_signal']['judge_preference_win_rate'], 0.0)
        self.assertFalse(scratch['reasoning_signal']['longer_trace_is_always_better'])
        self.assertEqual('reward_hacking', scratch['failure_modes']['primary_watch'])
        self.assertTrue(scratch['failure_modes']['length_bias_flag'])
        self.assertIn('artifacts/scratch-manual/rlhf_reasoning_reward.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('RLHF and reasoning RL reward signals', figure_text)
        self.assertIn('verifier bonus', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('tiny_numeric_reasoning_rl', framework['simulation'])
        self.assertEqual(4, framework['rollout_batch_size'])
        self.assertGreater(framework['policy_update']['reward_mean_after'], framework['policy_update']['reward_mean_before'])
        self.assertGreater(framework['policy_update']['advantage_mean_after'], framework['policy_update']['advantage_mean_before'])
        self.assertLessEqual(framework['policy_update']['kl_after'], framework['policy_update']['kl_guardrail'])
        self.assertIn('PPO', framework['policy_update']['update_family'])
        self.assertGreater(framework['reasoning_eval']['answer_accuracy_after'], framework['reasoning_eval']['answer_accuracy_before'])
        self.assertGreater(framework['reasoning_eval']['verifier_consistency_after'], framework['reasoning_eval']['verifier_consistency_before'])
        self.assertTrue(framework['reasoning_eval']['judge_length_bias_flag'])
        self.assertEqual('reward_hacking', framework['failure_mode_probes']['highest_risk'])
        self.assertGreater(framework['failure_mode_probes']['verbosity_delta'], 0.0)
        self.assertGreater(framework['failure_mode_probes']['over_refusal_delta'], 0.0)

        self.assertIn('# 06 RLHF and Reasoning RL 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('reward model', observed_text)
        self.assertIn('PPO-family', observed_text)
        self.assertIn('verifier', observed_text)
        self.assertIn('judge', observed_text)
        self.assertIn('reasoning RL', observed_text)
        self.assertIn('reward hacking', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
