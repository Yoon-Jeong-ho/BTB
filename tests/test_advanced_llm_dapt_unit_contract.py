from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '03_domain_adaptive_pretraining'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'dapt_tradeoff.svg'
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


class TestAdvancedLLMDAPTUnitContract(unittest.TestCase):
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

    def test_docs_are_korean_first_and_show_execution_examples(self) -> None:
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
        self.assertIn('python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/03_domain_adaptive_pretraining/analysis.py', readme)
        self.assertIn('domain shift', readme)
        self.assertIn('continued pretraining', readme)
        self.assertIn('catastrophic forgetting', readme)
        self.assertIn('replay mixture', readme)
        self.assertIn('data selection', readme)
        self.assertIn('stopping', readme)
        self.assertIn('dapt_tradeoff.svg', readme)

        self.assertIn('실행 결과 예시', theory)
        self.assertIn('domain shift', theory)
        self.assertIn('continued pretraining', theory)
        self.assertIn('catastrophic forgetting', theory)
        self.assertIn('replay mixture', theory)
        self.assertIn('data selection', theory)
        self.assertIn('stopping', theory)

        self.assertIn('학습자', reflection)
        self.assertIn('domain shift', reflection)
        self.assertIn('catastrophic forgetting', reflection)
        self.assertIn('replay mixture', reflection)
        self.assertIn('data selection', reflection)
        self.assertIn('stopping', reflection)

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
        self.assertIn('domain shift', text)
        self.assertIn('catastrophic forgetting', text)
        self.assertIn('replay mixture', text)
        self.assertIn('data selection', text)
        self.assertIn('stopping', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/03_domain_adaptive_pretraining/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/03_domain_adaptive_pretraining/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/03_domain_adaptive_pretraining/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/03_domain_adaptive_pretraining/analysis.py')
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

        self.assertEqual(set(scratch['strategies']), {'pure_domain', 'replay_mixture'})
        self.assertEqual(1.0, scratch['strategies']['pure_domain']['domain_share'])
        self.assertGreater(scratch['strategies']['replay_mixture']['general_share'], 0.0)
        self.assertGreater(
            scratch['strategies']['pure_domain']['in_domain_gain_final'],
            scratch['strategies']['replay_mixture']['in_domain_gain_final'],
        )
        self.assertGreater(
            scratch['strategies']['pure_domain']['general_regression_final'],
            scratch['strategies']['replay_mixture']['general_regression_final'],
        )
        self.assertEqual('pure_domain', scratch['comparison']['fastest_adapter'])
        self.assertEqual('replay_mixture', scratch['comparison']['safer_retention'])
        self.assertEqual('replay_mixture', scratch['comparison']['balanced_recommendation'])
        self.assertIn('artifacts/scratch-manual/dapt_tradeoff.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Domain-adaptive pretraining trade-offs', figure_text)
        self.assertIn('General retention', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(set(framework['strategies']), {'pure_domain', 'replay_mixture'})
        self.assertGreater(framework['base_losses']['domain'], framework['base_losses']['general'])
        self.assertGreater(
            framework['strategies']['pure_domain']['general_regression'],
            framework['strategies']['replay_mixture']['general_regression'],
        )
        self.assertLess(
            framework['strategies']['pure_domain']['final_domain_loss'],
            framework['base_losses']['domain'],
        )
        self.assertLess(
            framework['strategies']['replay_mixture']['final_domain_loss'],
            framework['base_losses']['domain'],
        )
        self.assertLess(
            framework['strategies']['replay_mixture']['final_general_loss'],
            framework['strategies']['pure_domain']['final_general_loss'],
        )
        self.assertGreater(framework['strategies']['pure_domain']['guardrail_exceeded_step'], 0)
        self.assertGreater(framework['strategies']['replay_mixture']['recommended_stop_step'], 0)
        self.assertEqual('curated_domain', framework['data_selection']['preferred'])
        self.assertGreater(
            framework['data_selection']['curated_domain']['selection_score'],
            framework['data_selection']['noisy_domain']['selection_score'],
        )

        self.assertIn('# 03 Domain Adaptive Pretraining 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('pure domain', observed_text)
        self.assertIn('replay mixture', observed_text)
        self.assertIn('domain shift', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
