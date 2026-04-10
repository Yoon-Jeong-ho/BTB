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

    def _track_map(self, data: dict[str, Any]) -> dict[str, Any]:
        tracks = data.get('tracks', {})
        if isinstance(tracks, dict):
            return tracks
        if isinstance(tracks, list):
            mapped: dict[str, Any] = {}
            for entry in tracks:
                if not isinstance(entry, dict):
                    continue
                name = entry.get('name') or entry.get('track') or entry.get('id')
                if isinstance(name, str):
                    mapped[name] = entry
            return mapped
        self.fail('curriculum_status.json must declare tracks as a dict or list')

    def _iter_unit_entries(self, track_name: str, track_data: Any):
        if not isinstance(track_data, dict):
            self.fail(f'track {track_name} must be an object')

        units = track_data.get('units', {})
        if isinstance(units, dict):
            for unit_name, unit_data in units.items():
                yield unit_name, unit_data
            return
        if isinstance(units, list):
            for entry in units:
                if not isinstance(entry, dict):
                    self.fail(f'track {track_name} units must be objects')
                unit_name = entry.get('name') or entry.get('id') or entry.get('slug')
                if not isinstance(unit_name, str):
                    self.fail(f'track {track_name} unit missing name/id/slug')
                yield unit_name, entry
            return
        self.fail(f'track {track_name} units must be a dict or list')

    def test_status_file_lists_expanded_tracks(self) -> None:
        track_map = self._track_map(self._load_status())

        for track in [
            '02_deep_learning',
            '05_advanced_nlp_llm',
            '06_training_systems',
            '07_frontier_labs',
        ]:
            self.assertIn(track, track_map)

    def test_declared_units_have_valid_status_and_readmes(self) -> None:
        track_map = self._track_map(self._load_status())

        for track_name, track_data in track_map.items():
            for unit_name, unit_data in self._iter_unit_entries(track_name, track_data):
                self.assertIsInstance(unit_data, dict, f'{track_name}/{unit_name} must be an object')
                status = unit_data.get('status')
                self.assertIn(status, VALID_STATUSES, f'{track_name}/{unit_name} has invalid status')
                self.assertTrue(
                    (ROOT / track_name / unit_name / 'README.md').exists(),
                    f'missing {track_name}/{unit_name}/README.md',
                )


if __name__ == '__main__':
    unittest.main()
