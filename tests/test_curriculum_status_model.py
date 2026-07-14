from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tests.test_curriculum_topology import CANONICAL_CURRICULUM_LADDER

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
EXPECTED_TRACKS = CANONICAL_CURRICULUM_LADDER
VALID_STATUSES = {'planned', 'outlined', 'runnable'}
VALID_FIDELITIES = {'concept-toy', 'framework-toy', 'real-data', 'gpu-capable'}
VALID_DIFFICULTIES = {'beginner', 'intermediate', 'advanced'}
VALID_COMPUTE = {'cpu', 'cpu-or-cuda', 'optional-multiprocess'}
UNIT_DIR_NAME_RE = re.compile(r'^\d+_')


class TestCurriculumStatusModel(unittest.TestCase):
    def _load_status(self) -> dict[str, Any]:
        path = ROOT / 'docs' / 'curriculum_status.json'
        self.assertTrue(path.exists(), 'missing docs/curriculum_status.json')
        return json.loads(path.read_text(encoding='utf-8'))

    def _discover_unit_dirs(self, track_name: str) -> set[str]:
        track_path = ROOT / track_name
        self.assertTrue(track_path.is_dir(), f'missing track directory {track_name}')
        return {
            child.name
            for child in track_path.iterdir()
            if child.is_dir()
            and UNIT_DIR_NAME_RE.match(child.name)
            and (child / 'README.md').is_file()
        }

    def test_status_file_lists_expanded_tracks(self) -> None:
        data = self._load_status()
        self.assertEqual(set(data.keys()), {'tracks'})
        tracks = data['tracks']
        self.assertIsInstance(tracks, dict, 'curriculum_status.json must use a dict-valued tracks manifest')
        self.assertEqual(set(tracks.keys()), set(EXPECTED_TRACKS))

    def test_manifest_exactly_matches_track_unit_directories(self) -> None:
        data = self._load_status()
        self.assertEqual(set(data.keys()), {'tracks'})
        tracks = data['tracks']
        self.assertIsInstance(tracks, dict, 'curriculum_status.json must use a dict-valued tracks manifest')
        self.assertEqual(set(tracks.keys()), set(EXPECTED_TRACKS))

        for track_name, units in tracks.items():
            self.assertIsInstance(track_name, str, 'track names must be strings')
            self.assertIsInstance(units, dict, f'{track_name} must map to a dict of unit statuses')
            declared_units = set(units.keys())
            discovered_units = self._discover_unit_dirs(track_name)
            self.assertEqual(
                declared_units,
                discovered_units,
                f'{track_name} manifest units must exactly match on-disk unit directories',
            )

    def test_declared_units_have_valid_status_and_readmes(self) -> None:
        data = self._load_status()
        self.assertEqual(set(data.keys()), {'tracks'})
        tracks = data['tracks']
        self.assertIsInstance(tracks, dict, 'curriculum_status.json must use a dict-valued tracks manifest')
        self.assertEqual(set(tracks.keys()), set(EXPECTED_TRACKS))

        for track_name, units in tracks.items():
            self.assertIsInstance(track_name, str, 'track names must be strings')
            self.assertIsInstance(units, dict, f'{track_name} must map to a dict of unit statuses')
            for unit_name, status in units.items():
                self.assertIsInstance(unit_name, str, f'{track_name} unit names must be strings')
                self.assertIsInstance(status, str, f'{track_name}/{unit_name} status must be a string')
                self.assertIn(status, VALID_STATUSES, f'{track_name}/{unit_name} has invalid status')
                self.assertTrue(
                    (ROOT / track_name / unit_name / 'README.md').exists(),
                    f'missing {track_name}/{unit_name}/README.md',
                )

    def test_declared_lesson_metadata_is_standard_yaml(self) -> None:
        data = self._load_status()
        lesson_paths = []
        for track_name, units in data['tracks'].items():
            for unit_name in units:
                lesson_path = ROOT / track_name / unit_name / 'lesson.yaml'
                if lesson_path.exists():
                    lesson_paths.append(lesson_path)

        self.assertTrue(lesson_paths, 'expected at least one lesson.yaml under declared units')
        for lesson_path in lesson_paths:
            with self.subTest(lesson=str(lesson_path.relative_to(ROOT))):
                parsed = yaml.safe_load(lesson_path.read_text(encoding='utf-8'))
                self.assertIsInstance(parsed, dict)
                self.assertIsInstance(parsed.get('objective'), str)
                for key in ['prereqs', 'key_terms', 'required_outputs', 'analysis_questions']:
                    if key in parsed:
                        self.assertIsInstance(parsed[key], list, f'{key} must be a list')

    def test_every_manifest_lesson_uses_runner_parser(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SCRIPTS_DIR)))
        from _lesson_metadata import load_lesson_metadata

        data = self._load_status()
        parsed_count = 0
        for track_name, units in data['tracks'].items():
            for unit_name in units:
                lesson_path = ROOT / track_name / unit_name / 'lesson.yaml'
                with self.subTest(lesson=str(lesson_path.relative_to(ROOT))):
                    metadata = load_lesson_metadata(lesson_path)
                    self.assertIsInstance(metadata.get('objective'), str)
                    parsed_count += 1

        self.assertEqual(48, parsed_count)

    def test_strict_curriculum_audit_reports_all_units_and_no_errors(self) -> None:
        result = subprocess.run(
            [sys.executable, 'scripts/audit_curriculum.py', '--strict'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(48, payload['unit_count'])
        self.assertEqual([], payload['errors'])
        self.assertIn('fidelity', payload['coverage'])
        self.assertIn('compute', payload['coverage'])

    def test_every_declared_unit_exposes_learner_facing_effort_metadata(self) -> None:
        data = self._load_status()

        for track_name, units in data['tracks'].items():
            for unit_name in units:
                lesson_path = ROOT / track_name / unit_name / 'lesson.yaml'
                self.assertTrue(lesson_path.is_file(), f'missing {lesson_path.relative_to(ROOT)}')
                parsed = yaml.safe_load(lesson_path.read_text(encoding='utf-8'))
                lesson = str(lesson_path.relative_to(ROOT))

                self.assertIn(parsed.get('fidelity'), VALID_FIDELITIES, f'{lesson} has invalid fidelity')
                self.assertIn(
                    parsed.get('difficulty'),
                    VALID_DIFFICULTIES,
                    f'{lesson} has invalid difficulty',
                )
                self.assertIsInstance(
                    parsed.get('estimated_minutes'),
                    int,
                    f'{lesson} estimated_minutes must be an integer',
                )
                self.assertGreater(
                    parsed['estimated_minutes'],
                    0,
                    f'{lesson} estimated_minutes must be positive',
                )
                self.assertIn(parsed.get('compute'), VALID_COMPUTE, f'{lesson} has invalid compute')

    def test_prerequisite_path_references_must_exist(self) -> None:
        sys.path.insert(0, str(SCRIPTS_DIR))
        self.addCleanup(lambda: sys.path.remove(str(SCRIPTS_DIR)))
        from audit_curriculum import _validate_prerequisite_references

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / '00_foundations/01_tensor_shapes').mkdir(parents=True)
            (root / 'docs').mkdir()
            (root / 'docs/existing.md').write_text('# ok\n', encoding='utf-8')
            errors = _validate_prerequisite_references(
                root,
                root / 'lesson.yaml',
                {
                    'prereqs': [
                        '00_foundations/01_tensor_shapes',
                        'docs/existing',
                        'docs/missing 선행 문서',
                        '일반/자연어 설명은 경로가 아님',
                    ]
                },
            )

        self.assertEqual(1, len(errors))
        self.assertIn('docs/missing', errors[0])


if __name__ == '__main__':
    unittest.main()
