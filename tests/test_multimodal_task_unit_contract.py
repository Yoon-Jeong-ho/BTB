from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_multimodal' / '01_image_text_retrieval'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_DIR / 'retrieval_heatmap.svg'
FRAMEWORK_METRICS = FRAMEWORK_DIR / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_DIR / 'latest_report.md'
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

GENERATED_DIRS = [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]


class TestMultimodalTaskUnitContract(unittest.TestCase):
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
        for directory in GENERATED_DIRS:
            if directory.exists():
                shutil.rmtree(directory)

    def _load_module(self, name: str, relative_path: str):
        path = ROOT / relative_path
        spec = importlib.util.spec_from_file_location(name, path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_readme_and_theory_are_korean_first_and_include_examples(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertRegex(readme, r'[가-힣]')
        self.assertRegex(theory, r'[가-힣]')
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('retrieval_heatmap.svg', readme)
        self.assertIn('실행 결과 예시', theory)
        self.assertIn('Recall@K', theory)
        self.assertIn('PyTorch', theory)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', lesson)
        self.assertIn('scratch svg figure', lesson)
        self.assertIn('analysis_questions:', lesson)
        self.assertIn('Recall@1', lesson)
        self.assertIn('hard negative', lesson)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_multimodal/01_image_text_retrieval/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_analysis_fails_when_required_metric_keys_are_missing(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        scratch_payload = {
            'image_to_text_recall_at_1': 1.0,
            'text_to_image_recall_at_1': 0.75,
            'text_to_image_recall_at_2': 1.0,
            'hardest_negative_pair': 'demo pair',
            # intentionally missing hardest_negative_similarity
        }
        framework_payload = {
            'image_to_text_recall_at_1': 1.0,
            'text_to_image_recall_at_1': 1.0,
            'symmetric_loss': 0.1,
            'logits_shape': [4, 4],
        }
        self._write_json(SCRATCH_METRICS, scratch_payload)
        self._write_json(FRAMEWORK_METRICS, framework_payload)

        result = self._run('05_multimodal/01_image_text_retrieval/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('metrics schema validation failed', error_text)
        self.assertIn('scratch metrics missing keys', error_text)
        self.assertIn('hardest_negative_similarity', error_text)

    def test_scratch_and_framework_validate_batch_size(self) -> None:
        scratch_lab = self._load_module(
            'multimodal_task_scratch_lab',
            '05_multimodal/01_image_text_retrieval/scratch_lab.py',
        )

        with self.assertRaisesRegex(ValueError, 'image/text batch size must match'):
            scratch_lab.retrieval_metrics(
                np.ones((4, 5), dtype=np.float64),
                np.ones((3, 5), dtype=np.float64),
                temperature=0.25,
            )

        if torch is None:
            self.skipTest('PyTorch not installed; skipping framework batch-size validation')

        framework_lab = self._load_module(
            'multimodal_task_framework_lab',
            '05_multimodal/01_image_text_retrieval/framework_lab.py',
        )
        with self.assertRaisesRegex(ValueError, 'image/text batch size must match'):
            framework_lab.compute_logits(
                torch.ones((4, 5), dtype=torch.float32),
                torch.ones((3, 9), dtype=torch.float32),
                temperature=0.2,
            )

    @unittest.skipIf(torch is None, 'PyTorch not installed; skipping framework run contract')
    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('05_multimodal/01_image_text_retrieval/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_multimodal/01_image_text_retrieval/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_multimodal/01_image_text_retrieval/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'analysis observed report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual(4, scratch['pair_count'])
        self.assertEqual([4, 4], scratch['similarity_matrix_shape'])
        self.assertEqual(1.0, scratch['image_to_text_recall_at_1'])
        self.assertEqual(0.75, scratch['text_to_image_recall_at_1'])
        self.assertEqual(1.0, scratch['text_to_image_recall_at_2'])
        self.assertGreater(scratch['mean_positive_similarity'], scratch['hardest_negative_similarity'])
        self.assertEqual('artifacts/scratch-manual/retrieval_heatmap.svg', scratch['figure_path'])
        self.assertIn('<svg', figure)
        self.assertIn('Image-text retrieval heatmap', figure)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual([4, 5], framework['image_input_shape'])
        self.assertEqual([4, 9], framework['text_input_shape'])
        self.assertEqual([4, 4], framework['logits_shape'])
        self.assertEqual(1.0, framework['image_to_text_recall_at_1'])
        self.assertEqual(1.0, framework['text_to_image_recall_at_1'])
        self.assertLess(framework['loss_history_tail'][-1], framework['loss_history_head'][0])
        self.assertLess(framework['symmetric_loss'], scratch['symmetric_contrastive_loss'])
        self.assertIn('ranked_matches', framework)

        self.assertIn('# 01 Image-Text Retrieval 실행 관측', observed)
        self.assertIn('## 한국어 해석', observed)
        self.assertIn('[THEORY.md](../../THEORY.md)', observed)
        self.assertIn('scratch text→image Recall@1', observed)
        self.assertIn('framework text→image Recall@1', observed)
        self.assertEqual(stable_before, analysis)
        self.assertIn('latest_report.md', analysis)
        self.assertIn('## 관련 이론', analysis)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis)


if __name__ == '__main__':
    unittest.main()
