from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '06_training_systems/04_fsdp_checkpointing_and_offload'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH = SCRATCH_DIR / 'metrics.json'
FRAMEWORK = FRAMEWORK_DIR / 'metrics.json'
SVG = SCRATCH_DIR / 'fsdp_memory_tradeoffs.svg'
OBSERVED = ANALYSIS_DIR / 'latest_report.md'
ANALYSIS = UNIT / 'analysis.md'
REQUIRED = [
    'README.md', 'THEORY.md', 'PREREQS.md', 'lesson.yaml',
    'scratch_lab.py', 'framework_lab.py', 'analysis.py', 'analysis.md',
    'reflection.md', 'artifacts',
]


class TestFSDPCheckpointingUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, rel: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, rel], cwd=ROOT, text=True, capture_output=True, check=False)

    def _cleanup(self) -> None:
        for d in [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]:
            if d.exists():
                shutil.rmtree(d)

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)
        self.assertEqual('', (ARTIFACTS / '.gitkeep').read_text(encoding='utf-8'))

    def test_docs_and_metadata_are_runnable_and_korean_first(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        prereqs = (UNIT / 'PREREQS.md').read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')

        for doc in [readme, theory, prereqs, reflection]:
            self.assertRegex(doc, r'[가-힣]')
        self.assertIn('Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertNotIn('Status: outlined', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertIn('status: runnable', lesson)
        self.assertIn('scratch_lab.py', lesson)
        self.assertIn('framework_lab.py', lesson)
        self.assertIn('analysis.py', lesson)
        self.assertIn('full state dict', lesson)
        self.assertIn('sharded state dict', lesson)
        self.assertIn('required_outputs:', lesson)
        self.assertIn('analysis_questions:', lesson)

    def test_analysis_requires_metrics_with_actionable_failure(self) -> None:
        self.addCleanup(self._cleanup)
        self._cleanup()
        result = self._run(str((UNIT / 'analysis.py').relative_to(ROOT)))
        combined = result.stdout + result.stderr
        self.assertNotEqual(0, result.returncode)
        self.assertIn('필수 metrics 파일이 없습니다', combined)
        self.assertIn('scratch_lab.py', combined)
        self.assertIn('framework_lab.py', combined)

    def test_labs_analysis_artifacts_and_stable_analysis(self) -> None:
        self.addCleanup(self._cleanup)
        self._cleanup()
        stable_before = ANALYSIS.read_text(encoding='utf-8')

        for script in ['scratch_lab.py', 'framework_lab.py', 'analysis.py']:
            result = self._run(str((UNIT / script).relative_to(ROOT)))
            self.assertEqual(0, result.returncode, result.stderr)

        self.assertTrue(SCRATCH.exists())
        self.assertTrue(FRAMEWORK.exists())
        self.assertTrue(SVG.exists())
        self.assertTrue(OBSERVED.exists())
        self.assertIn('<svg', SVG.read_text(encoding='utf-8'))

        scratch = json.loads(SCRATCH.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK.read_text(encoding='utf-8'))
        self.assertTrue(scratch['cpu_safe_simulation'])
        self.assertEqual('FULL_SHARD', scratch['sharding_strategy'])
        self.assertLess(scratch['fsdp_checkpointed_peak_gpu_mb'], scratch['fsdp_forward_peak_gpu_mb'])
        self.assertLess(scratch['cpu_offload_gpu_peak_mb'], scratch['fsdp_checkpointed_peak_gpu_mb'])
        self.assertIn('all_gather_full_params', scratch['lifecycle_events'])
        self.assertIn('reduce_scatter_gradients', scratch['lifecycle_events'])
        self.assertIn('backend', framework)
        self.assertEqual('cpu-simulated-fsdp-checkpoint-offload', framework['backend'])
        self.assertIn('full_state_dict', framework['state_dict_modes'])
        self.assertIn('sharded_state_dict', framework['state_dict_modes'])
        self.assertEqual('sharded_state_dict', framework['best_resume_mode_by_peak'])
        self.assertEqual(stable_before, ANALYSIS.read_text(encoding='utf-8'))
        observed = OBSERVED.read_text(encoding='utf-8')
        self.assertIn('## 한국어 해석', observed)
        self.assertIn('full state dict', observed)
        self.assertIn('sharded state dict', observed)

    def test_framework_simulation_is_deterministic(self) -> None:
        self.addCleanup(self._cleanup)
        self._cleanup()
        first = self._run(str((UNIT / 'framework_lab.py').relative_to(ROOT)))
        self.assertEqual(0, first.returncode, first.stderr)
        first_metrics = FRAMEWORK.read_text(encoding='utf-8')
        second = self._run(str((UNIT / 'framework_lab.py').relative_to(ROOT)))
        self.assertEqual(0, second.returncode, second.stderr)
        self.assertEqual(first_metrics, FRAMEWORK.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
