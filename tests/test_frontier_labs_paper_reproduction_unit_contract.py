from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '07_frontier_labs' / '01_paper_reproduction_playground'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_METRICS = ARTIFACTS / 'scratch-manual' / 'metrics.json'
SCRATCH_FIGURE = ARTIFACTS / 'scratch-manual' / 'paper_reproduction_matrix.svg'
FRAMEWORK_METRICS = ARTIFACTS / 'framework-manual' / 'metrics.json'
OBSERVED_REPORT = ARTIFACTS / 'analysis-manual' / 'latest_report.md'
ANALYSIS_MD = UNIT / 'analysis.md'

REQUIRED_FILES = [
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
GENERATED_FILES = [SCRATCH_METRICS, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT]
GENERATED_DIRS = [
    ARTIFACTS / 'scratch-manual',
    ARTIFACTS / 'framework-manual',
    ARTIFACTS / 'analysis-manual',
]


class TestFrontierLabsPaperReproductionUnitContract(unittest.TestCase):
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
        for path in GENERATED_FILES:
            if path.exists():
                path.unlink()
        for directory in GENERATED_DIRS:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    def test_unit_has_required_runnable_files(self) -> None:
        for rel in REQUIRED_FILES:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_docs_and_metadata_advertise_cpu_safe_runnable_reproduction_contract(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        analysis = ANALYSIS_MD.read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')

        self.assertIn('status: runnable', lesson)
        self.assertIn('cpu_safe: true', lesson)
        self.assertIn('deterministic: true', lesson)
        self.assertIn('claim/evidence matrix', lesson)
        self.assertIn('baseline/reported/reproduced comparison', lesson)
        self.assertIn('variance/mismatch hypotheses', lesson)
        self.assertIn('artifact hygiene checklist', lesson)

        self.assertRegex('\n'.join(readme.splitlines()[:10]), r'[가-힣]')
        self.assertIn('> Status: runnable', readme)
        self.assertIn('CPU-safe deterministic', readme)
        self.assertIn('실행 방법', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('paper_reproduction_matrix.svg', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)

        combined = readme + analysis + reflection
        for keyword in [
            'claim/evidence matrix',
            'baseline',
            'reported',
            'reproduced',
            'scope control',
            'variance',
            'mismatch hypothesis',
            'artifact hygiene',
        ]:
            self.assertIn(keyword, combined)

    def test_artifacts_gitkeep_is_locked(self) -> None:
        gitkeep = ARTIFACTS / '.gitkeep'
        self.assertTrue(gitkeep.exists())
        self.assertEqual('', gitkeep.read_text(encoding='utf-8'))

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        for path in GENERATED_FILES:
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('07_frontier_labs/01_paper_reproduction_playground/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_claim_level_reproduction_artifacts(self) -> None:
        for path in GENERATED_FILES:
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('07_frontier_labs/01_paper_reproduction_playground/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('07_frontier_labs/01_paper_reproduction_playground/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'observed report missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        figure = SCRATCH_FIGURE.read_text(encoding='utf-8')
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        stable_after = ANALYSIS_MD.read_text(encoding='utf-8')

        self.assertEqual('runnable', scratch['status'])
        self.assertTrue(scratch['cpu_safe'])
        self.assertEqual('claim_level_reproduction_playground', scratch['mode'])
        self.assertEqual(3, len(scratch['claim_evidence_matrix']))
        self.assertEqual(['claim_id', 'claim', 'evidence_type', 'acceptance_rule', 'observed_signal', 'decision'], scratch['claim_evidence_columns'])
        self.assertIn('scope control', scratch['scope_control']['principle'])
        self.assertEqual('reduced_claim', scratch['scope_control']['claim_scope'])
        self.assertEqual('classification_proxy_slice', scratch['scope_control']['dataset_scope'])
        self.assertEqual(3, len(scratch['comparisons']))
        c1 = scratch['comparisons']['C1_adapter_efficiency']
        self.assertEqual(0.842, c1['baseline']['accuracy'])
        self.assertEqual(0.851, c1['reported']['accuracy'])
        self.assertEqual(0.846, c1['reproduced']['accuracy'])
        self.assertAlmostEqual(0.004, c1['delta_vs_baseline'])
        self.assertLess(c1['delta_vs_reported'], 0.0)
        self.assertGreater(scratch['variance_summary']['accuracy_std'], 0.0)
        self.assertIn('preprocessing_alignment', scratch['mismatch_hypotheses'][0]['hypothesis_id'])
        self.assertIn('seed_variance', {item['hypothesis_id'] for item in scratch['mismatch_hypotheses']})
        self.assertEqual([], scratch['artifact_hygiene']['missing_required_artifacts'])
        self.assertTrue(scratch['artifact_hygiene']['ready_for_handoff'])
        self.assertIn('artifacts/scratch-manual/paper_reproduction_matrix.svg', scratch['artifacts']['figure'])
        self.assertIn('<svg', figure)
        self.assertIn('Claim/evidence reproduction matrix', figure)
        self.assertIn('C1_adapter_efficiency', figure)

        self.assertEqual('runnable', framework['status'])
        self.assertEqual('cpu_deterministic_reproduction_harness', framework['framework'])
        self.assertEqual('offline_no_network_no_paper_download', framework['runtime_contract']['network_policy'])
        self.assertTrue(framework['runtime_contract']['cpu_safe'])
        self.assertTrue(framework['runtime_contract']['deterministic'])
        self.assertEqual('claim_id -> evidence -> comparison -> mismatch_hypothesis -> artifact', framework['experiment_card_schema']['flow'])
        self.assertEqual('same_protocol_reproduced_baseline_vs_method', framework['comparison_policy']['primary_comparison'])
        self.assertIn('baseline', framework['comparison_policy']['comparison_layers'])
        self.assertIn('reported', framework['comparison_policy']['comparison_layers'])
        self.assertIn('reproduced', framework['comparison_policy']['comparison_layers'])
        self.assertEqual('review_before_capstone_handoff', framework['reproduction_decision']['decision'])
        self.assertIn('scope_boundary', framework['artifact_manifest']['required_files'])
        self.assertIn('claim_evidence_matrix', framework['artifact_manifest']['required_files'])
        self.assertIn('mismatch_hypotheses', framework['artifact_manifest']['required_files'])
        self.assertEqual([], framework['artifact_manifest']['missing'])

        self.assertIn('# 01 Paper Reproduction Playground 실행 관측', observed)
        self.assertIn('## claim/evidence matrix', observed)
        self.assertIn('## baseline / reported / reproduced 비교', observed)
        self.assertIn('## scope control', observed)
        self.assertIn('## variance와 mismatch hypothesis', observed)
        self.assertIn('## artifact hygiene', observed)
        self.assertIn('C1_adapter_efficiency', observed)
        self.assertIn('preprocessing_alignment', observed)
        self.assertIn('[THEORY.md](./THEORY.md)', observed)
        self.assertEqual(stable_before, stable_after)

    def test_script_stdout_is_reproducible(self) -> None:
        for path in GENERATED_FILES:
            self._preserve_path(path)
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        first = self._run('07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py')
        second = self._run('07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py')

        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)


if __name__ == '__main__':
    unittest.main()
