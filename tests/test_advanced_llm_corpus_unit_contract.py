from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '02_corpus_tokenizer_and_data_mixture'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'corpus_quality_overview.svg'
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


class TestAdvancedLlmCorpusUnitContract(unittest.TestCase):
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

    def test_readme_and_theory_are_korean_first_with_execution_examples(self) -> None:
        readme_text = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory_text = (UNIT / 'THEORY.md').read_text(encoding='utf-8')

        self.assertRegex('\n'.join(readme_text.splitlines()[:8]), r'[가-힣]')
        self.assertRegex('\n'.join(theory_text.splitlines()[:8]), r'[가-힣]')

        self.assertIn('Status: runnable', readme_text)
        self.assertIn('실행 결과 예시', readme_text)
        self.assertIn('scratch_lab.py', readme_text)
        self.assertIn('framework_lab.py', readme_text)
        self.assertIn('corpus_quality_overview.svg', readme_text)

        self.assertIn('실행 결과 예시', theory_text)
        self.assertIn('dedup', theory_text)
        self.assertIn('contamination', theory_text)
        self.assertIn('multilingual', theory_text)
        self.assertIn('token budget', theory_text)

    def test_lesson_metadata_mentions_required_outputs_and_questions(self) -> None:
        text = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertIn('status: runnable', text)
        self.assertIn('required_outputs:', text)
        self.assertIn('scratch svg figure', text)
        self.assertIn('stable analysis markdown', text)
        self.assertIn('analysis_questions:', text)
        self.assertIn('deduplication', text)
        self.assertIn('contamination', text)
        self.assertIn('multilingual mixture', text)
        self.assertIn('token budget', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/analysis.py')
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

        self.assertEqual(11, scratch['raw_document_count'])
        self.assertGreater(scratch['dedup_removed_documents'], 0)
        self.assertEqual(2, scratch['contamination_blocked_documents'])
        self.assertLess(
            scratch['tokenizers']['toy_whitespace']['avg_tokens_per_doc'],
            scratch['tokenizers']['toy_aggressive_subword']['avg_tokens_per_doc'],
        )
        self.assertLess(
            scratch['tokenizers']['toy_whitespace']['chars_per_token'],
            scratch['tokenizers']['toy_unigram_like']['chars_per_token'],
        )
        self.assertAlmostEqual(1.0, sum(scratch['mixture_token_share'].values()), places=6)
        self.assertEqual('artifacts/scratch-manual/corpus_quality_overview.svg', scratch['figure_path'])

        self.assertEqual('cpu', framework['device'])
        self.assertGreater(framework['removed_exact_duplicates'], 0)
        self.assertGreater(framework['removed_near_duplicates'], 0)
        self.assertEqual(2, framework['contamination_blocked'])
        self.assertAlmostEqual(1.0, sum(framework['language_token_share'].values()), places=6)
        self.assertAlmostEqual(1.0, sum(framework['domain_token_share'].values()), places=6)
        self.assertLess(
            framework['tokenizer_stats']['toy_unigram_like']['avg_tokens_per_doc'],
            framework['tokenizer_stats']['toy_aggressive_subword']['avg_tokens_per_doc'],
        )
        self.assertEqual(64, framework['token_budget']['context_window'])
        self.assertEqual(4, len(framework['batch_preview']))

        self.assertIn('<svg', figure_text)
        self.assertIn('Corpus quality and mixture overview', figure_text)
        self.assertIn('# 02 Corpus, Tokenizer, and Data Mixture 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
