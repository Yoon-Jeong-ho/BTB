from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / '07_frontier_labs' / '05_open_ended_research_tracks'
ARTIFACTS = UNIT / 'artifacts'
SCRATCH_DIR = ARTIFACTS / 'scratch-manual'
FRAMEWORK_DIR = ARTIFACTS / 'framework-manual'
ANALYSIS_DIR = ARTIFACTS / 'analysis-manual'
SCRATCH_METRICS = SCRATCH_DIR / 'metrics.json'
SCRATCH_FIGURE = SCRATCH_DIR / 'research_track_map.svg'
FRAMEWORK_METRICS = FRAMEWORK_DIR / 'metrics.json'
OBSERVED_REPORT = ANALYSIS_DIR / 'latest_report.md'
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

GENERATED_DIRS = [SCRATCH_DIR, FRAMEWORK_DIR, ANALYSIS_DIR]
RESEARCH_CONCEPTS = [
    'research scope',
    'north-star question',
    'hypothesis registry',
    'iteration boundary',
    'kill criteria',
    'evidence standard',
    'negative result',
    'inconclusive result',
    'stop',
    'pause',
    'escalate',
    'archive',
    'reopen condition',
]


class TestFrontierLabsOpenEndedResearchUnitContract(unittest.TestCase):
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
        self.assertTrue((ARTIFACTS / '.gitkeep').exists())
        self.assertEqual('', (ARTIFACTS / '.gitkeep').read_text(encoding='utf-8'))

    def test_docs_and_metadata_advertise_runnable_open_research_contract(self) -> None:
        lesson = (UNIT / 'lesson.yaml').read_text(encoding='utf-8')
        readme = (UNIT / 'README.md').read_text(encoding='utf-8')
        theory = (UNIT / 'THEORY.md').read_text(encoding='utf-8')
        reflection = (UNIT / 'reflection.md').read_text(encoding='utf-8')
        analysis = ANALYSIS_MD.read_text(encoding='utf-8')
        combined = '\n'.join([lesson, readme, theory, reflection, analysis])

        self.assertIn('status: runnable', lesson)
        self.assertIn('cpu_safe: true', lesson)
        self.assertIn('deterministic: true', lesson)
        self.assertIn('scratch_lab.py', lesson)
        self.assertIn('framework_lab.py', lesson)
        self.assertIn('analysis.py', lesson)
        self.assertIn('research_track_map.svg', lesson)

        self.assertRegex('\n'.join(readme.splitlines()[:10]), r'[가-힣]')
        self.assertIn('> Status: runnable', readme)
        self.assertIn('CPU-safe deterministic simulation', readme)
        self.assertIn('실행 방법', readme)
        self.assertIn('실행 결과 예시', readme)
        self.assertNotIn('sample shape only', readme)
        self.assertNotIn('후속 applied 단계', readme)
        self.assertNotIn('outlined 단계', readme)

        for concept in RESEARCH_CONCEPTS:
            self.assertIn(concept, combined)

    def test_analysis_requires_metrics_with_actionable_error(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        result = self._run('07_frontier_labs/05_open_ended_research_tracks/analysis.py')

        self.assertNotEqual(0, result.returncode)
        error_text = result.stdout + result.stderr
        self.assertIn('필수 metrics 파일이 없습니다', error_text)
        self.assertIn('먼저 scratch_lab.py와 framework_lab.py를 실행하세요', error_text)

    def test_labs_and_analysis_generate_deterministic_research_artifacts(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()
        stable_before = ANALYSIS_MD.read_text(encoding='utf-8')

        scratch_result = self._run('07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py')
        self.assertEqual(0, scratch_result.returncode, scratch_result.stderr)
        framework_result = self._run('07_frontier_labs/05_open_ended_research_tracks/framework_lab.py')
        self.assertEqual(0, framework_result.returncode, framework_result.stderr)
        analysis_result = self._run('07_frontier_labs/05_open_ended_research_tracks/analysis.py')
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
        self.assertEqual('frontier-open-ended-research-v1', scratch['track_id'])
        self.assertIn('north-star question', scratch['research_scope'])
        self.assertIn('research scope', scratch['research_scope'])
        self.assertIn('out_of_scope', scratch['research_scope'])
        self.assertGreaterEqual(len(scratch['research_scope']['out_of_scope']), 2)
        self.assertEqual('hypothesis registry', scratch['hypothesis_registry']['type'])
        self.assertGreaterEqual(len(scratch['hypothesis_registry']['hypotheses']), 4)

        result_types = {item['result_type'] for item in scratch['evidence_log']}
        self.assertIn('negative result', result_types)
        self.assertIn('inconclusive result', result_types)
        self.assertIn('trust failure', result_types)
        for hypothesis in scratch['hypothesis_registry']['hypotheses']:
            self.assertIn('iteration boundary', hypothesis)
            self.assertIn('kill criteria', hypothesis)
            self.assertIn('evidence standard', hypothesis)
            self.assertIn('reopen condition', hypothesis)

        self.assertIn('<svg', figure)
        self.assertIn('Open-ended research track map', figure)
        self.assertIn('hypothesis registry', figure)

        self.assertEqual('runnable', framework['status'])
        self.assertEqual('cpu_deterministic_open_research_ops_sim', framework['framework'])
        self.assertEqual({'stop', 'pause', 'escalate', 'archive'}, set(framework['decision_summary']['decision_counts']))
        decisions = {item['decision'] for item in framework['decision_log']}
        self.assertEqual({'stop', 'pause', 'escalate', 'archive'}, decisions)
        self.assertEqual('archive', framework['decision_by_result_type']['negative result'])
        self.assertEqual('pause', framework['decision_by_result_type']['inconclusive result'])
        self.assertEqual('escalate', framework['decision_by_result_type']['trust failure'])
        self.assertEqual('stop', framework['decision_by_result_type']['success stop'])
        self.assertIn('reopen condition', framework['archive_contract'])
        self.assertIn('evidence standard', framework['operation_contract']['required_fields'])
        self.assertIn('kill criteria', framework['operation_contract']['required_fields'])
        self.assertTrue(framework['operation_contract']['archive_every_iteration'])

        for phrase in [
            '# 05 Open-Ended Research Tracks 실행 관측',
            'research scope',
            'north-star question',
            'hypothesis registry',
            'iteration boundary',
            'kill criteria',
            'evidence standard',
            'negative result',
            'inconclusive result',
            'stop / pause / escalate / archive',
            'reopen condition',
            '[THEORY.md](./THEORY.md)',
        ]:
            self.assertIn(phrase, observed)

        self.assertEqual(stable_before, ANALYSIS_MD.read_text(encoding='utf-8'))

    def test_script_stdout_is_reproducible(self) -> None:
        self.addCleanup(self._cleanup_generated_outputs)
        self._cleanup_generated_outputs()

        first = self._run('07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py')
        second = self._run('07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py')

        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)


if __name__ == '__main__':
    unittest.main()
