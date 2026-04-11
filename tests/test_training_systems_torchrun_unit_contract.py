from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '06_training_systems/01_torchrun_and_ddp_basics'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH = ARTIFACTS / 'scratch-manual' / 'metrics.json'
FRAMEWORK = ARTIFACTS / 'framework-manual' / 'metrics.json'
OBSERVED = ARTIFACTS / 'analysis-manual' / 'latest_report.md'
ANALYSIS = UNIT / 'analysis.md'
REQUIRED = [
    'README.md', 'THEORY.md', 'PREREQS.md', 'lesson.yaml',
    'scratch_lab.py', 'framework_lab.py', 'analysis.py', 'analysis.md',
    'reflection.md', 'artifacts',
]

class TestTorchrunUnitContract(unittest.TestCase):
    maxDiff = None

    def _run(self, rel: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, rel], cwd=ROOT, text=True, capture_output=True, check=False)

    def _cleanup(self) -> None:
        for d in [ARTIFACTS / 'scratch-manual', ARTIFACTS / 'framework-manual', ARTIFACTS / 'analysis-manual']:
            if d.exists():
                shutil.rmtree(d)

    def test_unit_has_required_files(self) -> None:
        for rel in REQUIRED:
            self.assertTrue((UNIT / rel).exists(), rel)

    def test_docs_and_metadata(self) -> None:
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        self.assertRegex(readme, r'[가-힣]')
        self.assertRegex(theory, r'[가-힣]')
        self.assertIn('Status: runnable', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertIn('status: runnable', lesson)
        self.assertIn('required_outputs:', lesson)
        self.assertIn('analysis_questions:', lesson)
        self.assertEqual('', (ARTIFACTS / '.gitkeep').read_text(encoding='utf-8'))

    def test_analysis_requires_metrics(self) -> None:
        self.addCleanup(self._cleanup)
        self._cleanup()
        result = self._run(str((UNIT / 'analysis.py').relative_to(ROOT)))
        self.assertNotEqual(0, result.returncode)
        self.assertIn('필수 metrics 파일이 없습니다', result.stdout + result.stderr)

    def test_labs_and_analysis_generate_outputs(self) -> None:
        self.addCleanup(self._cleanup)
        self._cleanup()
        stable_before = ANALYSIS.read_text(encoding='utf-8')
        for script in ['scratch_lab.py', 'framework_lab.py', 'analysis.py']:
            result = self._run(str((UNIT / script).relative_to(ROOT)))
            self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(SCRATCH.exists())
        self.assertTrue(FRAMEWORK.exists())
        self.assertTrue(OBSERVED.exists())
        scratch = json.loads(SCRATCH.read_text(encoding='utf-8'))
        framework = json.loads(FRAMEWORK.read_text(encoding='utf-8'))
        self.assertIn('averaged_gradient', scratch)
        self.assertIn('backend', framework)
        self.assertEqual(stable_before, ANALYSIS.read_text(encoding='utf-8'))
        self.assertIn('## 한국어 해석', OBSERVED.read_text(encoding='utf-8'))

if __name__ == '__main__':
    unittest.main()
