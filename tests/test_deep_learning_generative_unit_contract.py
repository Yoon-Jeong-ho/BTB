from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '06_generative_models_vae_gan'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'vae_gan_diagnostics.svg'
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


class TestDeepLearningGenerativeUnitContract(unittest.TestCase):
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
        readme_text = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory_text = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertTrue(self._korean_first(readme_text))
        self.assertTrue(self._korean_first(theory_text))

        self.assertIn('> Status: runnable', readme_text)
        self.assertIn('실행 결과 예시', readme_text)
        self.assertIn('scratch_lab.py', readme_text)
        self.assertIn('framework_lab.py', readme_text)
        self.assertIn('vae_gan_diagnostics.svg', readme_text)

        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('reparameterization trick', theory_text)
        self.assertIn('adversarial', theory_text)
        self.assertIn('posterior collapse', theory_text)
        self.assertIn('mode collapse', theory_text)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('framework metrics json', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('reparameterization trick', text)
        self.assertIn('mode collapse', text)
        self.assertIn('posterior collapse', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/06_generative_models_vae_gan/analysis.py')

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

        scratch_result = self._run('02_deep_learning/06_generative_models_vae_gan/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/06_generative_models_vae_gan/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/06_generative_models_vae_gan/analysis.py')
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

        self.assertEqual(2, scratch['vae']['input_dim'])
        self.assertEqual(2, scratch['vae']['latent_dim'])
        self.assertEqual([8, 2], scratch['vae']['mu_shape'])
        self.assertEqual([8, 2], scratch['vae']['logvar_shape'])
        self.assertEqual([8, 2], scratch['vae']['reconstruction_shape'])
        self.assertEqual(5, scratch['vae']['interpolation_steps'])
        self.assertGreater(scratch['vae']['prior_sample_spread'], 0.0)
        self.assertTrue(scratch['vae']['posterior_collapse_probe']['collapse_detected'])
        self.assertLess(
            scratch['vae']['posterior_collapse_probe']['collapsed_latent_usage'],
            scratch['vae']['posterior_collapse_probe']['healthy_latent_usage'],
        )
        self.assertGreater(
            scratch['vae']['posterior_collapse_probe']['collapsed_reconstruction_mse'],
            scratch['vae']['posterior_collapse_probe']['healthy_reconstruction_mse'],
        )

        self.assertEqual(2, scratch['gan']['noise_dim'])
        self.assertEqual(4, scratch['gan']['balanced_mode_coverage'])
        self.assertEqual(1, scratch['gan']['collapsed_mode_coverage'])
        self.assertTrue(scratch['gan']['collapse_detected'])
        self.assertGreater(
            scratch['gan']['balanced_pairwise_distance_mean'],
            scratch['gan']['collapsed_pairwise_distance_mean'],
        )
        self.assertIn('artifacts/scratch-manual/vae_gan_diagnostics.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('VAE vs GAN diagnostics', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(2, framework['vae']['latent_dim'])
        self.assertEqual([8, 2], framework['vae']['mu_shape'])
        self.assertEqual([8, 2], framework['vae']['reconstruction_shape'])
        self.assertGreater(framework['vae']['prior_sample_spread'], 0.0)
        self.assertTrue(framework['vae']['collapsed_probe']['collapse_detected'])
        self.assertLess(
            framework['vae']['collapsed_probe']['collapsed_latent_usage_mean_abs'],
            framework['vae']['posterior_usage_mean_abs'],
        )

        self.assertEqual(2, framework['gan']['noise_dim'])
        self.assertEqual([8, 2], framework['gan']['sample_shape'])
        self.assertGreaterEqual(framework['gan']['mode_coverage'], 3)
        self.assertTrue(framework['gan']['collapsed_probe']['collapse_detected'])
        self.assertEqual(1, framework['gan']['collapsed_probe']['mode_coverage'])
        self.assertTrue(framework['gan']['loss_only_is_ambiguous'])

        self.assertIn('# 06 Generative Models: VAE, GAN 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)

        OBSERVED_REPORT.write_text('오래된 관측', encoding='utf-8')
        rerun_result = self._run('02_deep_learning/06_generative_models_vae_gan/analysis.py')
        self.assertEqual(0, rerun_result.returncode, rerun_result.stderr)
        stable_after = ANALYSIS_MD.read_text(encoding='utf-8')
        observed_after = OBSERVED_REPORT.read_text(encoding='utf-8')
        self.assertEqual(stable_before, stable_after)
        self.assertNotEqual('오래된 관측', observed_after)
        self.assertIn('# 06 Generative Models: VAE, GAN 실행 관측', observed_after)


if __name__ == '__main__':
    unittest.main()
