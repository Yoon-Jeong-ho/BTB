from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '02_deep_learning' / '05_autoencoders_and_representation_learning'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'autoencoder_bottleneck.svg'
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


class TestDeepLearningAutoencoderUnitContract(unittest.TestCase):
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
        self.assertIn('autoencoder_bottleneck.svg', readme_text)
        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('reconstruction objective', theory_text)
        self.assertIn('bottleneck', theory_text)
        self.assertIn('denoising', theory_text)
        self.assertIn('compression', theory_text)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('framework metrics json', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('reconstruction objective', text)
        self.assertIn('denoising', text)
        self.assertIn('compression', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('02_deep_learning/05_autoencoders_and_representation_learning/analysis.py')

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

        scratch_result = self._run('02_deep_learning/05_autoencoders_and_representation_learning/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_deep_learning/05_autoencoders_and_representation_learning/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_deep_learning/05_autoencoders_and_representation_learning/analysis.py')
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

        self.assertEqual(8, scratch['input_dim'])
        self.assertEqual(3, len(scratch['latent_preview']['sample_0']))
        self.assertEqual([1, 2, 3], scratch['bottleneck_dims_compared'])
        self.assertEqual('artifacts/scratch-manual/autoencoder_bottleneck.svg', scratch['figure_path'])
        self.assertIn('1', scratch['bottleneck_results'])
        self.assertIn('2', scratch['bottleneck_results'])
        self.assertIn('3', scratch['bottleneck_results'])
        self.assertLess(
            scratch['bottleneck_results']['3']['reconstruction_mse'],
            scratch['bottleneck_results']['1']['reconstruction_mse'],
        )
        self.assertLess(
            scratch['bottleneck_results']['2']['reconstruction_mse'],
            scratch['bottleneck_results']['1']['reconstruction_mse'],
        )
        self.assertTrue(scratch['denoising_variant']['denoising_improves_over_noisy_input'])
        self.assertLess(
            scratch['denoising_variant']['denoised_mse'],
            scratch['denoising_variant']['raw_noisy_mse'],
        )
        self.assertLess(scratch['compression_variant']['compression_ratio'], 1.0)
        self.assertIn('encoder', scratch['encoder_decoder_roles'])

        self.assertEqual('cpu', framework['device'])
        self.assertEqual(8, framework['input_dim'])
        self.assertEqual(scratch['sample_count'], framework['sample_count'])
        self.assertEqual(3, framework['compression_autoencoder']['latent_dim'])
        self.assertEqual(1, framework['narrow_bottleneck_autoencoder']['latent_dim'])
        self.assertEqual([framework['sample_count'], 3], framework['compression_autoencoder']['latent_shape'])
        self.assertEqual([framework['sample_count'], 8], framework['compression_autoencoder']['reconstruction_shape'])
        self.assertLess(
            framework['compression_autoencoder']['final_loss'],
            framework['narrow_bottleneck_autoencoder']['final_loss'],
        )
        self.assertLess(
            framework['denoising_autoencoder']['final_loss'],
            framework['denoising_autoencoder']['raw_noisy_baseline_loss'],
        )
        self.assertGreater(
            framework['compression_autoencoder']['parameter_count'],
            framework['narrow_bottleneck_autoencoder']['parameter_count'],
        )

        self.assertIn('<svg', figure_text)
        self.assertIn('Autoencoder bottleneck diagnostics', figure_text)
        self.assertIn('# 05 Autoencoders and Representation Learning 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](../../THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)

        OBSERVED_REPORT.write_text('오래된 관측', encoding='utf-8')
        rerun_result = self._run('02_deep_learning/05_autoencoders_and_representation_learning/analysis.py')
        self.assertEqual(0, rerun_result.returncode, rerun_result.stderr)
        stable_after = ANALYSIS_MD.read_text(encoding='utf-8')
        observed_after = OBSERVED_REPORT.read_text(encoding='utf-8')
        self.assertEqual(stable_before, stable_after)
        self.assertNotEqual('오래된 관측', observed_after)
        self.assertIn('# 05 Autoencoders and Representation Learning 실행 관측', observed_after)


if __name__ == '__main__':
    unittest.main()
