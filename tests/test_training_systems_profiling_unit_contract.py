from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '06_training_systems' / '09_profiling_monitoring_and_failure_recovery'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_DIR / 'profiling_timeline.svg'
FRAMEWORK_METRICS = FRAMEWORK_DIR / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_DIR / 'latest_report.md'

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


class TestTrainingSystemsProfilingUnitContract(unittest.TestCase):
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

    def test_metadata_and_docs_advertise_runnable_cpu_safe_contract(self) -> None:
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
        self.assertIn('profiling_timeline.svg', lesson)
        self.assertIn('monitoring snapshot', lesson)
        self.assertIn('recovery decision', lesson)

        self.assertRegex(readme, r'[가-힣]')
        self.assertIn('Status: runnable', readme)
        self.assertIn('CPU-safe deterministic simulation', readme)
        self.assertIn('실행 방법', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('profiling_timeline.svg', readme)
        self.assertIn('OOM', readme)
        self.assertIn('hang', readme)
        self.assertIn('divergence', readme)
        self.assertIn('checkpoint', readme)

        for keyword in ['throughput', 'step time', 'memory', 'heartbeat', 'failure', 'recovery']:
            self.assertIn(keyword, analysis + reflection)

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_deterministic_operational_artifacts(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        scratch_result = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/analysis.py')
        self.assertEqual(0, analysis_result.returncode, analysis_result.stderr)

        self.assertTrue(SCRATCH_METRICS.exists(), 'scratch metrics missing')
        self.assertTrue(SCRATCH_FIGURE.exists(), 'scratch figure missing')
        self.assertTrue(FRAMEWORK_METRICS.exists(), 'framework metrics missing')
        self.assertTrue(OBSERVED_REPORT.exists(), 'analysis report missing')

        scratch = json.loads(SCRATCH_METRICS.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK_METRICS.read_text(encoding='utf-8'))
        observed = OBSERVED_REPORT.read_text(encoding='utf-8')
        figure = SCRATCH_FIGURE.read_text(encoding='utf-8')

        self.assertEqual('runnable', scratch['status'])
        self.assertTrue(scratch['cpu_safe_simulation'])
        self.assertEqual(8, scratch['profile_window']['steps'])
        self.assertEqual(4, scratch['profile_window']['world_size'])
        self.assertEqual(100.0, round(sum(scratch['time_breakdown_pct'].values()), 6))
        self.assertGreater(scratch['step_time_ms']['p95'], scratch['step_time_ms']['p50'])
        self.assertLess(scratch['throughput']['observed_tokens_per_sec'], scratch['throughput']['baseline_tokens_per_sec'])
        self.assertIn('communication_wait', scratch['dominant_bottleneck'])
        self.assertGreater(scratch['memory_snapshot']['peak_reserved_mb'], scratch['memory_snapshot']['peak_allocated_mb'])
        self.assertIn('rank_2_heartbeat_lag', scratch['alerts'])
        self.assertIn('checkpoint_age_exceeds_target', scratch['alerts'])
        self.assertIn('<svg', figure)
        self.assertIn('Profiling timeline', figure)

        self.assertEqual('runnable', framework['status'])
        self.assertEqual('cpu_deterministic_monitoring_recovery_sim', framework['framework'])
        self.assertIn('throughput_tokens_per_sec', framework['monitoring_contract']['required_signals'])
        self.assertIn('gpu_memory_allocated_reserved', framework['monitoring_contract']['required_signals'])
        self.assertIn('per_rank_heartbeat', framework['monitoring_contract']['required_signals'])
        self.assertIn('checkpoint_freshness_minutes', framework['monitoring_contract']['required_signals'])
        self.assertEqual('hang_or_straggler', framework['failure_triage']['selected_incident']['classification'])
        self.assertEqual('retry_from_last_good_checkpoint', framework['recovery_decision']['action'])
        self.assertTrue(framework['recovery_decision']['post_resume_validation']['passed'])
        self.assertLessEqual(framework['retry_policy']['attempts_used'], framework['retry_policy']['max_attempts'])
        self.assertIn('optimizer_state', framework['checkpoint_manifest']['required_state'])
        self.assertIn('sampler_state', framework['checkpoint_manifest']['required_state'])

        self.assertIn('# 09 Profiling, Monitoring, and Failure Recovery 실행 관측', observed)
        self.assertIn('## 병목 진단', observed)
        self.assertIn('## Failure triage', observed)
        self.assertIn('## Recovery decision', observed)
        self.assertIn('communication_wait', observed)
        self.assertIn('rank_2_heartbeat_lag', observed)
        self.assertIn('retry_from_last_good_checkpoint', observed)

    def test_script_stdout_is_reproducible(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        first = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py')
        second = self._run('06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py')

        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)


if __name__ == '__main__':
    unittest.main()
