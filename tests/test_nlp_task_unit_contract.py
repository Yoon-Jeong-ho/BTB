from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

UNIT_SPECS = {
    'text_classification': {
        'unit': ROOT / '03_nlp' / '01_text_classification',
        'readme_terms': ['실행 결과 예시', 'metrics.json', 'bag-of-words'],
        'theory_terms': ['실행 결과 예시', 'bag-of-words', 'PyTorch'],
        'lesson_terms': ['scratch svg figure', 'analysis_questions:', 'macro F1', 'bag-of-words'],
        'scratch_figure': 'artifacts/scratch-manual/token_signal.svg',
        'analysis_heading': '# 01 Text Classification 실행 관측',
        'scratch_metric_keys': ['eval_accuracy', 'eval_macro_f1', 'class_priors', 'top_positive_tokens', 'top_negative_tokens'],
        'framework_metric_keys': ['num_classes', 'label_names', 'loss_history_head', 'prediction_rows'],
        'numeric_checks': {
            'scratch_train_size_min': 6,
            'scratch_primary_metric': 'eval_accuracy',
            'scratch_primary_metric_min': 0.5,
            'scratch_secondary_metric': 'eval_macro_f1',
            'scratch_secondary_metric_min': 0.5,
            'framework_train_size_min': 6,
            'framework_primary_metric': 'eval_accuracy',
            'framework_primary_metric_min': 0.5,
            'framework_secondary_metric': 'eval_macro_f1',
            'framework_secondary_metric_min': 0.5,
        },
    },
    'named_entity_recognition': {
        'unit': ROOT / '03_nlp' / '02_named_entity_recognition',
        'readme_terms': ['실행 결과 예시', 'metrics.json', 'BIO'],
        'theory_terms': ['실행 결과 예시', 'BIO', 'PyTorch'],
        'lesson_terms': ['scratch svg figure', 'analysis_questions:', 'entity-level F1', 'BIO'],
        'scratch_figure': 'artifacts/scratch-manual/label_distribution.svg',
        'analysis_heading': '# 02 Named Entity Recognition 실행 관측',
        'scratch_metric_keys': ['token_accuracy', 'entity_precision', 'entity_recall', 'entity_f1', 'label_counts', 'alignment_example'],
        'framework_metric_keys': ['num_labels', 'label_names', 'loss_history_head', 'prediction_rows'],
        'numeric_checks': {
            'scratch_train_size_min': 6,
            'scratch_primary_metric': 'token_accuracy',
            'scratch_primary_metric_min': 0.5,
            'scratch_secondary_metric': 'entity_f1',
            'scratch_secondary_metric_min': 0.4,
            'framework_train_size_min': 6,
            'framework_primary_metric': 'token_accuracy',
            'framework_primary_metric_min': 0.5,
            'framework_secondary_metric': 'entity_f1',
            'framework_secondary_metric_min': 0.4,
        },
    },
    'machine_reading_comprehension': {
        'unit': ROOT / '03_nlp' / '03_machine_reading_comprehension',
        'readme_terms': ['실행 결과 예시', 'metrics.json', 'span extraction'],
        'theory_terms': ['실행 결과 예시', 'span extraction', 'PyTorch'],
        'lesson_terms': ['scratch svg figure', 'analysis_questions:', 'exact match', 'span extraction'],
        'scratch_figure': 'artifacts/scratch-manual/answerability_breakdown.svg',
        'analysis_heading': '# 03 Machine Reading Comprehension 실행 관측',
        'scratch_metric_keys': ['eval_exact_match', 'eval_token_f1', 'answerable_accuracy', 'no_answer_threshold', 'prediction_rows'],
        'framework_metric_keys': ['embedding_dim', 'hidden_dim', 'loss_history_head', 'prediction_rows', 'answerable_accuracy'],
        'numeric_checks': {
            'scratch_train_size_min': 6,
            'scratch_primary_metric': 'eval_exact_match',
            'scratch_primary_metric_min': 0.5,
            'scratch_secondary_metric': 'eval_token_f1',
            'scratch_secondary_metric_min': 0.5,
            'framework_train_size_min': 6,
            'framework_primary_metric': 'eval_exact_match',
            'framework_primary_metric_min': 0.5,
            'framework_secondary_metric': 'eval_token_f1',
            'framework_secondary_metric_min': 0.5,
        },
    },
}

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


class TestNlpTaskUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, relative_path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, relative_path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def _generated_dirs(self, unit: Path) -> list[Path]:
        artifacts = unit / 'artifacts'
        return [
            artifacts / 'scratch-manual',
            artifacts / 'framework-manual',
            artifacts / 'analysis-manual',
        ]

    def _cleanup_generated_outputs(self, unit: Path) -> None:
        for directory in self._generated_dirs(unit):
            if directory.exists():
                shutil.rmtree(directory)

    def test_units_have_required_files(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            unit = spec['unit']
            with self.subTest(unit=spec_name):
                for rel in REQUIRED:
                    self.assertTrue((unit / rel).exists(), rel)

    def test_readme_and_theory_are_korean_first_and_include_examples(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            unit = spec['unit']
            readme = (unit / 'README.md').read_text(encoding='utf-8')
            theory = (unit / 'THEORY.md').read_text(encoding='utf-8')

            with self.subTest(unit=spec_name, file='README.md'):
                self.assertRegex(readme, r'[가-힣]')
                for term in spec['readme_terms']:
                    self.assertIn(term, readme)

            with self.subTest(unit=spec_name, file='THEORY.md'):
                self.assertRegex(theory, r'[가-힣]')
                for term in spec['theory_terms']:
                    self.assertIn(term, theory)

    def test_lesson_metadata_mentions_required_outputs(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            lesson = (spec['unit'] / 'lesson.yaml').read_text(encoding='utf-8')
            with self.subTest(unit=spec_name):
                self.assertIn('required_outputs:', lesson)
                for term in spec['lesson_terms']:
                    self.assertIn(term, lesson)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            gitkeep = spec['unit'] / 'artifacts' / '.gitkeep'
            with self.subTest(unit=spec_name):
                self.assertTrue(gitkeep.exists())
                self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            unit = spec['unit']
            relative = unit.relative_to(ROOT)
            self.addCleanup(self._cleanup_generated_outputs, unit)
            self._cleanup_generated_outputs(unit)

            result = self._run(str(relative / 'analysis.py'))
            error_text = result.stdout + result.stderr

            with self.subTest(unit=spec_name):
                self.assertNotEqual(0, result.returncode)
                self.assertIn('필수 metrics 파일이 없습니다', error_text)
                self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_expected_outputs(self) -> None:
        for spec_name, spec in UNIT_SPECS.items():
            unit = spec['unit']
            artifacts = unit / 'artifacts'
            scratch_dir = artifacts / 'scratch-manual'
            framework_dir = artifacts / 'framework-manual'
            analysis_dir = artifacts / 'analysis-manual'
            scratch_metrics_path = scratch_dir / 'metrics.json'
            scratch_figure_path = unit / spec['scratch_figure']
            framework_metrics_path = framework_dir / 'metrics.json'
            observed_report_path = analysis_dir / 'latest_report.md'
            analysis_md_path = unit / 'analysis.md'
            relative = unit.relative_to(ROOT)
            numeric_checks = spec['numeric_checks']

            self.addCleanup(self._cleanup_generated_outputs, unit)
            self._cleanup_generated_outputs(unit)

            scratch_result = self._run(str(relative / 'scratch_lab.py'))
            framework_result = self._run(str(relative / 'framework_lab.py'))
            analysis_result = self._run(str(relative / 'analysis.py'))

            with self.subTest(unit=spec_name, phase='commands'):
                self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
                self.assertEqual(0, framework_result.returncode, framework_result.stderr)
                self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

            with self.subTest(unit=spec_name, phase='files'):
                self.assertTrue(scratch_metrics_path.exists(), 'scratch metrics missing')
                self.assertTrue(scratch_figure_path.exists(), 'scratch svg figure missing')
                self.assertTrue(framework_metrics_path.exists(), 'framework metrics missing')
                self.assertTrue(observed_report_path.exists(), 'analysis observed report missing')
                self.assertTrue(analysis_md_path.exists(), 'analysis.md missing')

            scratch = json.loads(scratch_metrics_path.read_text(encoding='utf-8'))
            framework = json.loads(framework_metrics_path.read_text(encoding='utf-8'))
            observed = observed_report_path.read_text(encoding='utf-8')
            analysis = analysis_md_path.read_text(encoding='utf-8')

            with self.subTest(unit=spec_name, phase='scratch-metrics'):
                self.assertEqual(spec['scratch_figure'], scratch['figure_path'])
                self.assertGreaterEqual(scratch['train_size'], numeric_checks['scratch_train_size_min'])
                self.assertGreaterEqual(scratch[numeric_checks['scratch_primary_metric']], numeric_checks['scratch_primary_metric_min'])
                self.assertGreaterEqual(scratch[numeric_checks['scratch_secondary_metric']], numeric_checks['scratch_secondary_metric_min'])
                for key in spec['scratch_metric_keys']:
                    self.assertIn(key, scratch)
                self.assertIn('<svg', scratch_figure_path.read_text(encoding='utf-8'))

            with self.subTest(unit=spec_name, phase='framework-metrics'):
                self.assertGreaterEqual(framework['train_size'], numeric_checks['framework_train_size_min'])
                self.assertGreaterEqual(framework[numeric_checks['framework_primary_metric']], numeric_checks['framework_primary_metric_min'])
                self.assertGreaterEqual(framework[numeric_checks['framework_secondary_metric']], numeric_checks['framework_secondary_metric_min'])
                for key in spec['framework_metric_keys']:
                    self.assertIn(key, framework)
                self.assertGreater(framework['vocab_size'], 5)
                self.assertGreater(len(framework['label_names']), 1)

            with self.subTest(unit=spec_name, phase='analysis'):
                self.assertIn(spec['analysis_heading'], observed)
                self.assertIn('## 한국어 해석', observed)
                self.assertIn('[THEORY.md](./THEORY.md)', observed)
                self.assertIn('latest_report.md', analysis)
                self.assertIn('## 관련 이론', analysis)
                self.assertIn('[THEORY.md](./THEORY.md)', analysis)
                self.assertNotIn(str(framework[numeric_checks['framework_primary_metric']]), analysis)


if __name__ == '__main__':
    unittest.main()
