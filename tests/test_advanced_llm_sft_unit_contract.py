from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '04_instruction_tuning_and_sft'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'sft_template_loss.svg'
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


class TestAdvancedLLMSFTUnitContract(unittest.TestCase):
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
        self.assertIn('python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/analysis.py', readme)
        self.assertIn('instruction format', readme)
        self.assertIn('supervised fine-tuning', readme)
        self.assertIn('input-output template', readme)
        self.assertIn('system', readme)
        self.assertIn('user', readme)
        self.assertIn('assistant', readme)
        self.assertIn('imitation', readme)
        self.assertIn('helpfulness', readme)
        self.assertIn('sft_template_loss.svg', readme)

        self.assertIn('실행 결과 예시', theory)
        self.assertIn('instruction format', theory)
        self.assertIn('supervised fine-tuning', theory)
        self.assertIn('input-output template', theory)
        self.assertIn('role framing', theory)
        self.assertIn('system', theory)
        self.assertIn('user', theory)
        self.assertIn('assistant', theory)
        self.assertIn('imitation', theory)
        self.assertIn('helpfulness', theory)

        self.assertIn('학습자', reflection)
        self.assertIn('instruction format', reflection)
        self.assertIn('system', reflection)
        self.assertIn('assistant', reflection)
        self.assertIn('imitation', reflection)
        self.assertIn('helpfulness', reflection)

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
        self.assertIn('instruction format', text)
        self.assertIn('supervised fine-tuning', text)
        self.assertIn('input-output template', text)
        self.assertIn('role framing', text)
        self.assertIn('system/user/assistant', text)
        self.assertIn('imitation', text)
        self.assertIn('helpfulness', text)
        self.assertIn('preference optimization', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/04_instruction_tuning_and_sft/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/04_instruction_tuning_and_sft/analysis.py')
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

        self.assertEqual('04_instruction_tuning_and_sft', scratch['setup']['unit'])
        self.assertEqual(set(scratch['template_views']), {'plain_instruction', 'chat_template'})
        self.assertEqual(['system', 'user', 'assistant'], scratch['template_views']['chat_template']['roles'])
        self.assertEqual('assistant_response_only', scratch['template_views']['chat_template']['target_region'])
        self.assertGreater(scratch['template_views']['plain_instruction']['total_tokens'], 0)
        self.assertGreater(scratch['loss_masking']['prompt_tokens_masked_out'], 0)
        self.assertGreater(scratch['loss_masking']['assistant_loss_tokens'], 0)
        self.assertGreater(
            scratch['loss_masking']['full_sequence_loss_tokens'],
            scratch['loss_masking']['assistant_loss_tokens'],
        )
        self.assertGreater(scratch['role_framing']['system_constraint_delta'], 0.0)
        self.assertEqual('chat_template', scratch['role_framing']['recommended_for_role_control'])
        self.assertGreater(scratch['imitation_vs_helpfulness']['canned_response_risk'], 0.0)
        self.assertGreater(
            scratch['imitation_vs_helpfulness']['format_imitation_score'],
            scratch['imitation_vs_helpfulness']['helpfulness_proxy_score'],
        )
        self.assertIn('artifacts/scratch-manual/sft_template_loss.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Instruction tuning and SFT', figure_text)
        self.assertIn('Assistant loss tokens', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('deterministic_numeric_sft', framework['framework'])
        self.assertEqual(4, framework['dataset_size'])
        self.assertEqual([4, framework['max_sequence_length']], framework['batch_shape']['input_ids'])
        self.assertEqual([4, framework['max_sequence_length']], framework['batch_shape']['labels'])
        self.assertEqual([4, framework['max_sequence_length']], framework['batch_shape']['loss_mask'])
        self.assertEqual(0, framework['loss_mask_summary']['prompt_loss_tokens'])
        self.assertGreater(framework['loss_mask_summary']['assistant_loss_tokens'], 0)
        self.assertGreater(framework['loss_mask_summary']['masked_prompt_tokens'], 0)
        self.assertLess(framework['training_curve'][-1]['assistant_loss'], framework['training_curve'][0]['assistant_loss'])
        self.assertGreater(
            framework['training_curve'][-1]['template_adherence'],
            framework['training_curve'][0]['template_adherence'],
        )
        self.assertGreater(framework['imitation_vs_helpfulness']['over_imitation_risk'], 0.0)
        self.assertEqual('preference_optimization_needed', framework['next_step']['why_sft_is_not_enough'])

        self.assertIn('# 04 Instruction Tuning and SFT 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('instruction format', observed_text)
        self.assertIn('supervised fine-tuning', observed_text)
        self.assertIn('system/user/assistant', observed_text)
        self.assertIn('imitation', observed_text)
        self.assertIn('helpfulness', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
