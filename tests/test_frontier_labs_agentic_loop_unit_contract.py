from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '07_frontier_labs' / '03_agentic_training_and_eval_loops'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_TRACE = SCRATCH_DIR / 'iteration_trace.jsonl'
SCRATCH_FIGURE = SCRATCH_DIR / 'agentic_loop_trace.svg'
FRAMEWORK_METRICS = FRAMEWORK_DIR / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_DIR / 'latest_report.md'
OBSERVED_JSON = ANALYSIS_DIR / 'observed_summary.json'

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

GENERATED_DIRS = [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]


class TestFrontierLabsAgenticLoopUnitContract(unittest.TestCase):
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

    def test_unit_has_required_runnable_files(self) -> None:
        for rel in REQUIRED_FILES:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_metadata_and_docs_advertise_cpu_safe_agentic_contract(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        analysis = (UNIT / 'analysis.md').read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')

        self.assertIn('status: runnable', lesson)
        self.assertIn('cpu_safe: true', lesson)
        self.assertIn('deterministic: true', lesson)
        self.assertIn('scratch_lab.py', lesson)
        self.assertIn('framework_lab.py', lesson)
        self.assertIn('analysis.py', lesson)
        self.assertIn('experiment contract', lesson)
        self.assertIn('planner', lesson)
        self.assertIn('executor', lesson)
        self.assertIn('verifier', lesson)
        self.assertIn('critic', lesson)
        self.assertIn('benchmark drift', lesson)

        self.assertRegex(readme, r'[가-힣]')
        self.assertIn('Status: runnable', readme)
        self.assertIn('CPU-safe deterministic simulation', readme)
        self.assertIn('실행 방법', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('agentic_loop_trace.svg', readme)
        self.assertIn('retry budget', readme)
        self.assertIn('stop rule', readme)
        self.assertIn('escalation rule', readme)

        for keyword in [
            'experiment contract',
            'planner',
            'executor',
            'verifier',
            'critic',
            'protocol match',
            'artifact completeness',
            'evidence bundle',
            'benchmark drift',
        ]:
            self.assertIn(keyword, analysis + reflection)

    def test_analysis_requires_prior_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_deterministic_agentic_loop_artifacts(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        scratch_result = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        for path in [SCRATCH_METRICS, SCRATCH_TRACE, SCRATCH_FIGURE, FRAMEWORK_METRICS, OBSERVED_REPORT, OBSERVED_JSON]:
            self.assertTrue(path.exists(), f'{path} missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        trace_lines = [json.loads(line) for line in SCRATCH_TRACE.read_text(encoding='utf-8').splitlines() if line.strip()]
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        observed_json = json.loads(OBSERVED_JSON.read_text(encoding='utf-8'))
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        figure = SCRATCH_FIGURE.read_text(encoding='utf-8')

        self.assertEqual('runnable', scratch['status'])
        self.assertTrue(scratch['cpu_safe_simulation'])
        self.assertEqual('agentic-train-eval-v1', scratch['loop_id'])
        self.assertEqual(3, scratch['experiment_contract']['retry_budget'])
        self.assertEqual('agentic_retrieval_eval', scratch['experiment_contract']['task'])
        self.assertIn('same_eval_split', scratch['experiment_contract']['frozen_constraints'])
        self.assertEqual(['planner', 'executor', 'verifier', 'critic'], scratch['role_sequence'])
        self.assertEqual(4, len(scratch['iterations']))
        self.assertEqual(4, len(trace_lines))
        self.assertEqual('escalate_to_human', scratch['final_decision']['action'])
        self.assertIn('benchmark_drift', scratch['final_decision']['reasons'])
        self.assertIn('<svg', figure)
        self.assertIn('Agentic loop trace', figure)

        by_iteration = {item['iteration']: item for item in scratch['iterations']}
        self.assertTrue(by_iteration[1]['verifier']['protocol_match'])
        self.assertFalse(by_iteration[2]['verifier']['protocol_match'])
        self.assertFalse(by_iteration[2]['verifier']['artifact_complete'])
        self.assertIn('rollback', by_iteration[2]['critic']['verdict'])
        self.assertEqual('stop_and_escalate', by_iteration[4]['critic']['verdict'])
        self.assertGreaterEqual(by_iteration[4]['verifier']['benchmark_drift_score'], 0.15)

        self.assertEqual('runnable', framework['status'])
        self.assertEqual('cpu_deterministic_agentic_loop_contract', framework['framework'])
        self.assertEqual(['planner', 'executor', 'verifier', 'critic'], framework['role_contract']['separation_order'])
        self.assertIn('planner_does_not_approve_own_plan', framework['role_contract']['anti_self_approval_rules'])
        self.assertEqual(3, framework['retry_policy']['max_retries'])
        self.assertEqual(2, framework['retry_policy']['attempts_used'])
        self.assertIn('same_failure_twice', framework['stop_rules'])
        self.assertIn('benchmark_drift_above_0.12', framework['escalation_rules'])
        self.assertTrue(framework['gate_summary']['protocol_match_required'])
        self.assertTrue(framework['gate_summary']['artifact_completeness_required'])
        self.assertEqual('needs_human_review', framework['gate_summary']['final_gate'])
        self.assertIn('config_hash', framework['evidence_bundle']['required_fields'])
        self.assertIn('verifier_gate', framework['evidence_bundle']['required_fields'])
        self.assertIn('critic_triage', framework['evidence_bundle']['required_fields'])
        self.assertGreaterEqual(framework['benchmark_drift']['observed_score'], 0.15)

        self.assertEqual('needs_human_review', observed_json['final_gate'])
        self.assertIn('benchmark_drift', observed_json['dominant_risk'])
        self.assertEqual(4, observed_json['iteration_count'])
        self.assertIn('# 03 Agentic Training and Eval Loops 실행 관측', observed)
        self.assertIn('## 역할 분리', observed)
        self.assertIn('## Gate verdict', observed)
        self.assertIn('## Benchmark drift', observed)
        self.assertIn('escalate_to_human', observed)

    def test_script_stdout_is_reproducible(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        first = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/scratch_lab.py')
        second = self._run('07_frontier_labs/03_agentic_training_and_eval_loops/scratch_lab.py')

        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)


if __name__ == '__main__':
    unittest.main()
