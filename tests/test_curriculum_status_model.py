from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from tests.test_curriculum_topology import CANONICAL_CURRICULUM_LADDER

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRACKS = CANONICAL_CURRICULUM_LADDER
VALID_STATUSES = {'planned', 'outlined', 'runnable'}


class TestCurriculumStatusModel(unittest.TestCase):
    def _load_status(self) -> dict[str, Any]:
        path = ROOT / 'docs' / 'curriculum_status.json'
        self.assertTrue(path.exists(), 'missing docs/curriculum_status.json')
        return json.loads(path.read_text(encoding='utf-8'))

    def _discover_unit_dirs(self, track_name: str) -> set[str]:
        track_path = ROOT / track_name
        self.assertTrue(track_path.is_dir(), f'missing track directory {track_name}')
        return {child.name for child in track_path.iterdir() if child.is_dir()}

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


if __name__ == '__main__':
    unittest.main()
