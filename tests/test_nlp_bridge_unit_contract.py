from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / '02_nlp_bridge'

UNIT = BRIDGE_ROOT / '01_tokenization_and_embeddings'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
FRAMEWORK_METRICS = ARTIFACTS / 'framework-manual' / 'metrics.json'
OBSERVED_REPORT = ARTIFACTS / 'analysis-manual' / 'latest_report.md'
ANALYSIS_MD = UNIT / 'analysis.md'
REQUIRED = [
    'README.md',
    'THEORY.md',
    'PREREQS.md',
    'lesson.yaml',
    'tokenization_fixture.py',
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

UNIT2 = BRIDGE_ROOT / '02_attention_and_transformer_block'
ARTIFACTS2 = UNIT2 / 'artifacts'
SCRATCH_METRICS2 = ARTIFACTS2 / 'scratch-manual' / 'metrics.json'
FRAMEWORK_METRICS2 = ARTIFACTS2 / 'framework-manual' / 'metrics.json'
OBSERVED_REPORT2 = ARTIFACTS2 / 'analysis-manual' / 'latest_report.md'
ANALYSIS_MD2 = UNIT2 / 'analysis.md'
REQUIRED2 = [
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
GENERATED_DIRS2 = [
    ARTIFACTS2 / 'scratch-manual',
    ARTIFACTS2 / 'framework-manual',
    ARTIFACTS2 / 'analysis-manual',
]


class TestNlpBridgeUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, relative_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _cleanup_generated_outputs(
        self,
        generated_files: tuple[Path, ...],
        generated_dirs: tuple[Path, ...],
    ) -> None:
        for path in generated_files:
            if path.exists():
                path.unlink()
        for directory in generated_dirs:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    def test_bridge_readme_links_bridge_units(self) -> None:
        text = (BRIDGE_ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertIn('01_tokenization_and_embeddings', text)
        self.assertIn('02_attention_and_transformer_block', text)
        self.assertIn('attention', text)
        self.assertIn('transformer', text)
        self.assertRegex(text, r'[가-힣]')

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_lesson_metadata_mentions_bridge_outputs(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('padding mask', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        generated_files = (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT)
        generated_dirs = tuple(GENERATED_DIRS)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        result = self._run('02_nlp_bridge/01_tokenization_and_embeddings/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_analysis_handles_empty_examples_and_no_unknowns(self) -> None:
        generated_files = (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT)
        generated_dirs = tuple(GENERATED_DIRS)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        (ARTIFACTS / 'scratch-manual').mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / 'framework-manual').mkdir(parents=True, exist_ok=True)
        SCRATCH_METRICS.write_text(
            json.dumps(
                {
                    'vocab_size': 26,
                    'examples': [],
                    'total_word_count': 0,
                    'total_subword_count': 0,
                    'subword_expansion_ratio': 0.0,
                    'unknown_token_total': 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        FRAMEWORK_METRICS.write_text(
            json.dumps(
                {
                    'input_ids_shape': [0, 0],
                    'embedded_shape': [0, 0, 0],
                    'padding_mask_shape': [0, 0],
                    'pooled_shape': [0, 0],
                    'non_pad_counts': [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

        result = self._run('02_nlp_bridge/01_tokenization_and_embeddings/analysis.py')

        self.assertEqual(0, result.returncode, result.stderr)
        observed_text = OBSERVED_REPORT.read_text(encoding='utf-8')
        self.assertIn('분석할 첫 예문이 없어', observed_text)
        self.assertIn('`[UNK]`가 관측되지 않았다', observed_text)

    def test_framework_lab_handles_empty_encoded_examples(self) -> None:
        generated_files = (FRAMEWORK_METRICS,)
        generated_dirs = (ARTIFACTS / 'framework-manual',)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        sys.path.insert(0, str(UNIT))
        self.addCleanup(lambda: sys.path.remove(str(UNIT)))

        spec = importlib.util.spec_from_file_location(
            'framework_lab_empty_case', UNIT / 'framework_lab.py'
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_build = module.build_encoded_examples
        module.build_encoded_examples = lambda: []
        self.addCleanup(setattr, module, 'build_encoded_examples', original_build)

        module.run()

        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing for empty case')
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        self.assertEqual(0, framework['batch_size'])
        self.assertEqual([0, 0], framework['input_ids_shape'])
        self.assertEqual([0, 0, 6], framework['embedded_shape'])
        self.assertEqual([0, 0], framework['padding_mask_shape'])
        self.assertEqual([0, 6], framework['pooled_shape'])
        self.assertEqual([], framework['sentence_order'])
        self.assertEqual([], framework['sequence_lengths_with_special_tokens'])
        self.assertEqual([], framework['non_pad_counts'])
        self.assertEqual([], framework['first_sentence_first_token_preview'])

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        generated_files = (SCRATCH_METRICS, FRAMEWORK_METRICS, OBSERVED_REPORT)
        generated_dirs = tuple(GENERATED_DIRS)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        scratch_result = self._run('02_nlp_bridge/01_tokenization_and_embeddings/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_nlp_bridge/01_tokenization_and_embeddings/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_nlp_bridge/01_tokenization_and_embeddings/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'observed analysis report missing')
        self.assertTrue(ANALYSIS_MD.exists(), 'analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        observed_text = OBSERVED_REPORT.read_text(encoding='utf-8')
        analysis_text = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual(3, len(scratch['examples']))
        self.assertGreater(scratch['subword_expansion_ratio'], 1.0)
        self.assertGreaterEqual(scratch['unknown_token_total'], 1)
        self.assertEqual([3, framework['max_sequence_length']], framework['input_ids_shape'])
        self.assertEqual([3, framework['max_sequence_length'], framework['embedding_dim']], framework['embedded_shape'])
        self.assertEqual([3, framework['max_sequence_length']], framework['padding_mask_shape'])
        self.assertEqual(
            [example['sentence'] for example in scratch['examples']],
            framework['sentence_order'],
        )
        self.assertEqual(
            [example['sequence_length_with_special_tokens'] for example in scratch['examples']],
            framework['sequence_lengths_with_special_tokens'],
        )
        self.assertEqual(
            max(example['sequence_length_with_special_tokens'] for example in scratch['examples']),
            framework['max_sequence_length'],
        )
        self.assertEqual(
            framework['sequence_lengths_with_special_tokens'],
            framework['non_pad_counts'],
        )
        self.assertTrue(framework['pad_token_row_is_zero'])
        self.assertEqual(0.0, framework['pad_vector_abs_max'])
        self.assertIn('# 01 Tokenization and Embeddings 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)
        self.assertNotIn(str(framework['first_sentence_first_token_preview'][0]), analysis_text)

    def test_second_unit_has_required_files(self) -> None:
        for rel in REQUIRED2:
            self.assertTrue((UNIT2 / rel).exists(), rel)

    def test_second_unit_lesson_metadata_mentions_attention_outputs(self) -> None:
        text = (UNIT2 / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('required_outputs:', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('padding mask', text)
        self.assertIn('causal mask', text)
        self.assertIn('transformer block', text)

    def test_second_unit_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS2 / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_second_unit_analysis_requires_metrics_with_actionable_error(self) -> None:
        generated_files = (SCRATCH_METRICS2, FRAMEWORK_METRICS2, OBSERVED_REPORT2)
        generated_dirs = tuple(GENERATED_DIRS2)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        result = self._run('02_nlp_bridge/02_attention_and_transformer_block/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_second_unit_labs_and_analysis_generate_expected_outputs(self) -> None:
        generated_files = (SCRATCH_METRICS2, FRAMEWORK_METRICS2, OBSERVED_REPORT2)
        generated_dirs = tuple(GENERATED_DIRS2)
        self.addCleanup(self._cleanup_generated_outputs, generated_files, generated_dirs)
        self._cleanup_generated_outputs(generated_files, generated_dirs)

        scratch_result = self._run('02_nlp_bridge/02_attention_and_transformer_block/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('02_nlp_bridge/02_attention_and_transformer_block/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('02_nlp_bridge/02_attention_and_transformer_block/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS2.exists(), 'second-unit scratch metrics missing')
        self.assertTrue(FRAMEWORK_METRICS2.exists(), 'second-unit framework metrics missing')
        self.assertTrue(OBSERVED_REPORT2.exists(), 'second-unit observed analysis report missing')
        self.assertTrue(ANALYSIS_MD2.exists(), 'second-unit analysis.md missing')

        scratch = json.loads(SCRATCH_METRICS2.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS2.read_text(encoding='utf-8'))
        observed_text = OBSERVED_REPORT2.read_text(encoding='utf-8')
        analysis_text = ANALYSIS_MD2.read_text(encoding='utf-8')

        self.assertEqual(4, scratch['sequence_length'])
        self.assertEqual(4, len(scratch['tokens']))
        self.assertEqual(4, len(scratch['raw_scores']))
        self.assertEqual(4, len(scratch['attention_weights']))
        self.assertEqual(4, len(scratch['strongest_links']))
        self.assertEqual('좋아해요', scratch['focus_query_token'])
        for row_sum in scratch['row_sums']:
            self.assertAlmostEqual(1.0, row_sum, places=5)
        self.assertIn('가중합', scratch['sequence_mixing_explanation'])

        self.assertEqual([3, 5], framework['input_ids_shape'])
        self.assertEqual([3, 5, 8], framework['embedded_shape'])
        self.assertEqual([3, 5], framework['key_padding_mask_shape'])
        self.assertEqual([5, 5], framework['causal_mask_shape'])
        self.assertEqual([3, 5, 8], framework['attention_output_shape'])
        self.assertEqual([3, 2, 5, 5], framework['attention_weights_shape'])
        self.assertEqual([3, 5, 8], framework['transformer_block_output_shape'])
        self.assertEqual([4, 4, 5], framework['valid_token_counts'])
        self.assertTrue(framework['pad_token_row_is_zero'])
        self.assertLessEqual(framework['pad_key_attention_max'], 1e-6)
        self.assertLessEqual(framework['future_attention_max'], 1e-6)
        self.assertEqual(5, len(framework['first_head_last_query_weights_batch0']))
        self.assertEqual(4, len(framework['first_token_block_preview']))

        self.assertIn('# 02 Attention and Transformer Block 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
