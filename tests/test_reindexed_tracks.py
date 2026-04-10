from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUTURE_TRACK_ROOTS = [
    '02_deep_learning',
    '03_nlp_bridge',
    '04_nlp',
    '05_advanced_nlp_llm',
    '06_training_systems',
    '07_frontier_labs',
    '08_multimodal_bridge',
    '09_multimodal',
]
RETIRED_TRACK_ROOTS = [
    '02_nlp_bridge',
    '03_nlp',
    '04_multimodal_bridge',
    '05_multimodal',
]


class TestReindexedTracks(unittest.TestCase):
    def test_reindexed_track_readmes_exist(self) -> None:
        for rel in [f'{track}/README.md' for track in FUTURE_TRACK_ROOTS]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_old_track_roots_are_gone(self) -> None:
        for rel in RETIRED_TRACK_ROOTS:
            self.assertFalse((ROOT / rel).exists(), rel)

    def test_user_facing_docs_stop_referencing_retired_roots(self) -> None:
        required_tokens = [
            '03_nlp_bridge',
            '04_nlp',
            '08_multimodal_bridge',
            '09_multimodal',
        ]

        for rel in ['README.md', 'docs/00_program_map.md', 'scripts/README.md']:
            text = (ROOT / rel).read_text(encoding='utf-8')
            for token in required_tokens:
                self.assertIn(token, text, f'{rel} missing {token}')
            for token in RETIRED_TRACK_ROOTS:
                pattern = rf'(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])'
                self.assertNotRegex(text, pattern, f'{rel} still mentions retired root {token}')


if __name__ == '__main__':
    unittest.main()
