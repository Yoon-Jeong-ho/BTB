from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestReindexedTracks(unittest.TestCase):
    def test_reindexed_track_readmes_exist(self) -> None:
        for rel in [
            '02_deep_learning/README.md',
            '03_nlp_bridge/README.md',
            '04_nlp/README.md',
            '05_advanced_nlp_llm/README.md',
            '06_training_systems/README.md',
            '07_frontier_labs/README.md',
            '08_multimodal_bridge/README.md',
            '09_multimodal/README.md',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_old_track_roots_are_gone(self) -> None:
        for rel in [
            '02_nlp_bridge',
            '03_nlp',
            '04_multimodal_bridge',
            '05_multimodal',
        ]:
            self.assertFalse((ROOT / rel).exists(), rel)


if __name__ == '__main__':
    unittest.main()
