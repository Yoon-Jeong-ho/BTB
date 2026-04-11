from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '05_advanced_nlp_llm' / '07_retrieval_augmented_generation_and_eval'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'rag_grounding_eval.svg'
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


class TestAdvancedLLMRAGUnitContract(unittest.TestCase):
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
        self.assertIn('python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/scratch_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/framework_lab.py', readme)
        self.assertIn('python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/analysis.py', readme)
        self.assertIn('rag_grounding_eval.svg', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)

        for text in (readme, theory, reflection):
            self.assertIn('retriever-reader', text)
            self.assertIn('retriever-generator', text)
            self.assertIn('retrieval grounding', text)
            self.assertIn('context injection', text)
            self.assertIn('citation', text)
            self.assertIn('evidence', text)
            self.assertIn('unsupported claim', text)
            self.assertIn('failure mode', text)
            self.assertIn('eval harness', text)
            self.assertIn('metrics', text)

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
        self.assertIn('retriever-reader', text)
        self.assertIn('retriever-generator', text)
        self.assertIn('retrieval grounding', text)
        self.assertIn('context injection', text)
        self.assertIn('citation/evidence expectation', text)
        self.assertIn('failure modes', text)
        self.assertIn('eval harness', text)
        self.assertIn('offline/online metric split', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/analysis.py')

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

        scratch_result = self._run('05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/analysis.py')
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

        self.assertEqual('07_retrieval_augmented_generation_and_eval', scratch['setup']['unit'])
        self.assertTrue(scratch['setup']['cpu_safe'])
        self.assertEqual(4, scratch['retrieval_batch']['query_count'])
        self.assertEqual(6, scratch['retrieval_batch']['chunk_count'])
        self.assertEqual(3, scratch['retrieval_batch']['top_k'])
        self.assertGreaterEqual(scratch['retrieval_metrics']['recall_at_3'], scratch['retrieval_metrics']['recall_at_1'])
        self.assertGreater(scratch['retrieval_metrics']['mrr'], 0.0)
        self.assertGreater(scratch['grounding_eval']['groundedness'], 0.0)
        self.assertLess(scratch['grounding_eval']['unsupported_claim_rate'], 0.5)
        self.assertEqual('claim-level evidence, not citation count', scratch['grounding_eval']['grounding_expectation'])
        self.assertEqual('retriever_reader', scratch['split_view']['lower_unsupported_claims'])
        self.assertEqual('retriever_generator', scratch['split_view']['higher_fluency'])
        self.assertIn('unsupported_claim', scratch['failure_modes']['primary_watch'])
        self.assertIn('missing_evidence', scratch['failure_modes']['observed_failure_modes'])
        self.assertIn('stale_source', scratch['failure_modes']['observed_failure_modes'])
        self.assertTrue(scratch['context_injection']['metadata_included'])
        self.assertTrue(scratch['context_injection']['citation_tags_required'])
        self.assertIn('artifacts/scratch-manual/rag_grounding_eval.svg', scratch['figure_path'])
        self.assertIn('<svg', figure_text)
        self.assertIn('Toy RAG grounding and retrieval metrics', figure_text)
        self.assertIn('retrieval recall@3', figure_text)
        self.assertIn('groundedness', figure_text)

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('deterministic_lightweight_rag', framework['simulation'])
        self.assertEqual([4, framework['embedding_dim']], framework['batch_shapes']['query_embeddings'])
        self.assertEqual([6, framework['embedding_dim']], framework['batch_shapes']['doc_embeddings'])
        self.assertEqual([4, 3], framework['batch_shapes']['topk_indices'])
        self.assertGreaterEqual(framework['retrieval_metrics']['recall_at_3'], framework['retrieval_metrics']['recall_at_1'])
        self.assertGreater(framework['retrieval_metrics']['ndcg_at_3'], 0.0)
        self.assertGreater(framework['answer_metrics']['groundedness'], framework['answer_metrics']['unsupported_claim_rate'])
        self.assertGreater(framework['answer_metrics']['citation_precision'], 0.0)
        self.assertTrue(framework['context_injection']['metadata_included'])
        self.assertTrue(framework['context_injection']['stale_source_penalty_enabled'])
        self.assertEqual('unsupported_claim', framework['failure_mode_probes']['highest_risk'])
        self.assertGreater(framework['eval_harness']['offline']['retriever_recall_at_3'], 0.0)
        self.assertGreater(framework['eval_harness']['online']['acceptance_proxy'], 0.0)
        self.assertGreater(framework['eval_harness']['online']['correction_rate_proxy'], 0.0)

        self.assertIn('# 07 Retrieval-Augmented Generation and Eval 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('retriever-reader', observed_text)
        self.assertIn('retriever-generator', observed_text)
        self.assertIn('retrieval grounding', observed_text)
        self.assertIn('context injection', observed_text)
        self.assertIn('citation', observed_text)
        self.assertIn('unsupported claim', observed_text)
        self.assertIn('eval harness', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
