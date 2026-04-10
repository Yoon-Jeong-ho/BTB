from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests.test_curriculum_topology import (
    CANONICAL_CURRICULUM_LADDER,
    assert_tokens_appear_in_order,
)

ROOT = Path(__file__).resolve().parents[1]


class TestCurriculumExpansionDocs(unittest.TestCase):
    def test_study_guide_mentions_expanded_tracks_in_order(self) -> None:
        path = ROOT / 'docs' / '02_study_guide.md'
        self.assertTrue(path.exists(), 'missing docs/02_study_guide.md')
        text = path.read_text(encoding='utf-8')
        assert_tokens_appear_in_order(self, text, CANONICAL_CURRICULUM_LADDER)

    def test_track_migration_map_mentions_required_renames(self) -> None:
        path = ROOT / 'docs' / '03_track_migration_map.md'
        self.assertTrue(path.exists(), 'missing docs/03_track_migration_map.md')
        text = path.read_text(encoding='utf-8')

        for old, new in [
            ('02_nlp_bridge', '03_nlp_bridge'),
            ('03_nlp', '04_nlp'),
            ('04_multimodal_bridge', '08_multimodal_bridge'),
            ('05_multimodal', '09_multimodal'),
        ]:
            pattern = rf"(?m)^.*{re.escape(old)}.*->.*{re.escape(new)}.*$"
            self.assertRegex(text, pattern)


if __name__ == '__main__':
    unittest.main()
