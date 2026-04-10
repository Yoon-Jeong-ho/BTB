from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VALID_STATUSES = {'planned', 'outlined', 'runnable'}


class TestCurriculumStatusModel(unittest.TestCase):
    def _load_status(self) -> dict[str, Any]:
        path = ROOT / 'docs' / 'curriculum_status.json'
        self.assertTrue(path.exists(), 'missing docs/curriculum_status.json')
        return json.loads(path.read_text(encoding='utf-8'))

    def test_status_file_lists_expanded_tracks(self) -> None:
        data = self._load_status()
        self.assertEqual(set(data.keys()), {'tracks'})
        tracks = data['tracks']
        self.assertIsInstance(tracks, dict, 'curriculum_status.json must use a dict-valued tracks manifest')

        for track in [
            '02_deep_learning',
            '05_advanced_nlp_llm',
            '06_training_systems',
            '07_frontier_labs',
        ]:
            self.assertIn(track, tracks)

    def test_declared_units_have_valid_status_and_readmes(self) -> None:
        data = self._load_status()
        self.assertEqual(set(data.keys()), {'tracks'})
        tracks = data['tracks']
        self.assertIsInstance(tracks, dict, 'curriculum_status.json must use a dict-valued tracks manifest')

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
