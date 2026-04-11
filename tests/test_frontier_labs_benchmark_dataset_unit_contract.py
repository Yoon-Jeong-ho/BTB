from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '07_frontier_labs' / '04_benchmark_and_dataset_construction'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'benchmark_dataset_overview.svg'
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


class TestFrontierLabsBenchmarkDatasetUnitContract(unittest.TestCase):
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

    def test_docs_are_korean_first_and_runnable(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        prereqs = (UNIT / 'PREREQS.md').read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')

        for text in (readme, theory, prereqs, reflection):
            self.assertRegex('\n'.join(text.splitlines()[:8]), r'[가-힣]')

        self.assertIn('> Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('python 07_frontier_labs/04_benchmark_and_dataset_construction/scratch_lab.py', readme)
        self.assertIn('python 07_frontier_labs/04_benchmark_and_dataset_construction/framework_lab.py', readme)
        self.assertIn('python 07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py', readme)
        self.assertIn('benchmark_dataset_overview.svg', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)
        self.assertNotIn('sample shape only', readme)

        for text in (readme, theory, reflection):
            self.assertIn('task contract', text)
            self.assertIn('dataset schema', text)
            self.assertIn('source/split manifest', text)
            self.assertIn('annotation rubric', text)
            self.assertIn('QC', text)
            self.assertIn('leakage', text)
            self.assertIn('contamination', text)
            self.assertIn('drift', text)
            self.assertIn('benchmark card', text)
            self.assertIn('versioning', text)
            self.assertIn('report template', text)

    def test_lesson_metadata_mentions_runnable_outputs_and_core_questions(self) -> None:
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
        self.assertIn('task contract', text)
        self.assertIn('dataset schema', text)
        self.assertIn('annotation QC', text)
        self.assertIn('leakage / contamination / drift audit', text)
        self.assertIn('benchmark versioning', text)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_benchmark_dataset_contract(self) -> None:
        for path in (SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT):
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('07_frontier_labs/04_benchmark_and_dataset_construction/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('07_frontier_labs/04_benchmark_and_dataset_construction/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py')
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

        self.assertEqual('04_benchmark_and_dataset_construction', scratch['setup']['unit'])
        self.assertTrue(scratch['setup']['cpu_safe'])
        self.assertEqual('btb-agent-benchmark-v1', scratch['benchmark_card']['benchmark_id'])
        self.assertIn('known_non_goals', scratch['benchmark_card'])
        self.assertEqual('agent_task_record', scratch['task_contract']['unit_of_record'])
        self.assertIn('tool_grounding', scratch['task_contract']['claim_boundaries'])
        self.assertIn('policy_compliance', scratch['task_contract']['rubric_dimensions'])
        self.assertIn('record_id', scratch['dataset_schema']['required_fields'])
        self.assertIn('slice_tags', scratch['dataset_schema']['required_fields'])
        self.assertIn('license_tier', scratch['dataset_schema']['required_fields'])
        self.assertEqual(12, scratch['source_manifest']['raw_records'])
        self.assertEqual(10, scratch['source_manifest']['accepted_records'])
        self.assertEqual(1, scratch['source_manifest']['excluded_by_license'])
        self.assertEqual(1, scratch['source_manifest']['excluded_by_schema'])
        self.assertEqual({'dev', 'test_private', 'test_public'}, set(scratch['split_manifest']['counts']))
        self.assertTrue(scratch['split_manifest']['source_disjoint'])
        self.assertTrue(scratch['split_manifest']['template_family_disjoint'])
        self.assertEqual(0, scratch['leakage_contamination_drift_audit']['exact_cross_split_overlap_hits'])
        self.assertEqual(1, scratch['leakage_contamination_drift_audit']['near_duplicate_review_flags'])
        self.assertEqual(2, scratch['leakage_contamination_drift_audit']['contamination_flags'])
        self.assertGreaterEqual(scratch['annotation_qc']['double_label_rate'], 0.3)
        self.assertGreaterEqual(scratch['annotation_qc']['agreement_score'], 0.75)
        self.assertEqual('expert_adjudication_if_major_disagreement', scratch['annotation_qc']['adjudication_rule'])
        self.assertEqual('2026-04-12', scratch['versioning']['frozen_on'])
        self.assertIn('known_limits', scratch['report_template']['sections'])
        self.assertIn('artifacts/scratch-manual/benchmark_dataset_overview.svg', scratch['figure_path'])

        self.assertEqual('cpu', framework['device'])
        self.assertEqual('deterministic_benchmark_dataset_pipeline', framework['simulation'])
        self.assertEqual('btb-agent-benchmark-v1', framework['benchmark_card']['benchmark_id'])
        self.assertEqual(10, framework['dataset_size'])
        self.assertEqual(['dev', 'test_public', 'test_private'], framework['splits'])
        self.assertTrue(framework['split_manifest']['source_disjoint'])
        self.assertTrue(framework['split_manifest']['template_family_disjoint'])
        self.assertEqual(0, framework['audit']['exact_cross_split_overlap_hits'])
        self.assertEqual(1, framework['audit']['near_duplicate_review_flags'])
        self.assertEqual(2, framework['audit']['contamination_flags'])
        self.assertEqual(2, len(framework['audit']['drift_watchlist']))
        self.assertIn('task_success', framework['annotation']['rubric_dimensions'])
        self.assertIn('groundedness', framework['annotation']['rubric_dimensions'])
        self.assertIn('policy_compliance', framework['annotation']['rubric_dimensions'])
        self.assertGreaterEqual(framework['annotation']['qc']['agreement_score'], 0.75)
        self.assertEqual('v1.0.0', framework['versioning']['version'])
        self.assertFalse(framework['versioning']['historically_comparable_to_v0'])
        self.assertIn('report_template', framework)
        self.assertIn('contamination_audit', framework['report_template']['sections'])

        self.assertIn('<svg', figure_text)
        self.assertIn('Benchmark dataset construction overview', figure_text)
        self.assertIn('task contract', figure_text)
        self.assertIn('annotation QC', figure_text)
        self.assertIn('# 04 Benchmark and Dataset Construction 실행 관측', observed_text)
        self.assertIn('## 한국어 해석', observed_text)
        self.assertIn('task contract', observed_text)
        self.assertIn('dataset schema', observed_text)
        self.assertIn('source/split manifest', observed_text)
        self.assertIn('annotation rubric', observed_text)
        self.assertIn('leakage', observed_text)
        self.assertIn('contamination', observed_text)
        self.assertIn('drift', observed_text)
        self.assertIn('benchmark card', observed_text)
        self.assertIn('versioning', observed_text)
        self.assertIn('report template', observed_text)
        self.assertIn('[THEORY.md](./THEORY.md)', observed_text)
        self.assertEqual(stable_before, analysis_text)
        self.assertIn('latest_report.md', analysis_text)
        self.assertIn('## 관련 이론', analysis_text)
        self.assertIn('[THEORY.md](./THEORY.md)', analysis_text)


if __name__ == '__main__':
    unittest.main()
