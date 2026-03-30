from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / '04_multimodal_bridge'
UNIT = BRIDGE_ROOT / '01_contrastive_alignment'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'alignment_heatmap.svg'
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


class TestMultimodalBridgeUnitContract(unittest.TestCase):
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
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            if path.exists():
                path.unlink()
        for directory in GENERATED_DIRS:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    def _load_module(self, name: str, relative_path: str):
        path = ROOT / relative_path
        spec = importlib.util.spec_from_file_location(name, path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_bridge_readme_links_first_unit_in_korean(self) -> None:
        text = (BRIDGE_ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('01_contrastive_alignment', text)
        self.assertIn('contrastive alignment', text.lower())
        self.assertRegex(text, r'[가-힣]')

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_readme_and_theory_include_execution_examples(self) -> None:
        readme_text = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory_text = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        self.assertIn('실행 결과 예시', readme_text)
        self.assertIn('alignment_heatmap.svg', readme_text)
        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('joint embedding space', theory_text)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('temperature', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('04_multimodal_bridge/01_contrastive_alignment/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_scratch_validates_batch_size_and_escapes_svg_labels(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        scratch_lab = self._load_module(
            'multimodal_bridge_scratch_lab',
            '04_multimodal_bridge/01_contrastive_alignment/scratch_lab.py',
        )

        with self.assertRaisesRegex(ValueError, 'image/text batch size must match'):
            scratch_lab.contrastive_metrics(
                np.ones((3, 3), dtype=np.float64),
                np.ones((2, 3), dtype=np.float64),
                temperature=0.2,
            )

        scratch_lab.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        scratch_lab.save_heatmap_svg(
            np.eye(2, dtype=np.float64),
            ['img <A&B>', 'txt > C'],
        )
        svg_text = SCRATCH_FIGURE.read_text(encoding='utf-8')
        self.assertIn('img &lt;A&amp;B&gt;', svg_text)
        self.assertIn('txt &gt; C', svg_text)
        self.assertNotIn('img <A&B>', svg_text)

    def test_framework_validates_batch_size(self) -> None:
        framework_lab = self._load_module(
            'multimodal_bridge_framework_lab',
            '04_multimodal_bridge/01_contrastive_alignment/framework_lab.py',
        )

        with self.assertRaisesRegex(ValueError, 'image/text batch size must match'):
            framework_lab.compute_logits(
                torch.ones((3, 3), dtype=torch.float32),
                torch.ones((2, 3), dtype=torch.float32),
                temperature=0.2,
            )

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('04_multimodal_bridge/01_contrastive_alignment/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('04_multimodal_bridge/01_contrastive_alignment/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('04_multimodal_bridge/01_contrastive_alignment/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'observed analysis report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure_text = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed_text = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis_text = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual(3, scratch['pair_count'])
        self.assertEqual([3, 3], scratch['similarity_matrix_shape'])
        self.assertEqual(1.0, scratch['top1_alignment_accuracy'])
        self.assertGreater(scratch['mean_positive_similarity'], scratch['mean_negative_similarity'])
        self.assertIn('artifacts/scratch-manual/alignment_heatmap.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Contrastive alignment heatmap', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual([3, 3], framework['logits_shape'])
        self.assertEqual([3], framework['labels_shape'])
        self.assertEqual(1.0, framework['top1_alignment_accuracy'])
        self.assertLess(framework['max_row_probability_sum_error'], 1e-6)
        self.assertAlmostEqual(scratch['symmetric_contrastive_loss'], framework['symmetric_loss'], places=6)

        self.assertIn('# 01 Contrastive Alignment 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
