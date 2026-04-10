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
ACTIVE_MOVED_CURRICULUM_DOCS = [
    '03_nlp_bridge/README.md',
    '03_nlp_bridge/01_tokenization_and_embeddings/README.md',
    '03_nlp_bridge/02_attention_and_transformer_block/README.md',
    '04_nlp/README.md',
    '04_nlp/01_text_classification/README.md',
    '04_nlp/01_text_classification/PREREQS.md',
    '04_nlp/02_named_entity_recognition/README.md',
    '04_nlp/02_named_entity_recognition/PREREQS.md',
    '04_nlp/03_machine_reading_comprehension/README.md',
    '04_nlp/03_machine_reading_comprehension/PREREQS.md',
    '08_multimodal_bridge/README.md',
    '08_multimodal_bridge/01_contrastive_alignment/README.md',
    '08_multimodal_bridge/01_contrastive_alignment/PREREQS.md',
    '08_multimodal_bridge/01_contrastive_alignment/reflection.md',
    '09_multimodal/README.md',
    '09_multimodal/01_image_text_retrieval/README.md',
    '09_multimodal/02_image_captioning/README.md',
    '09_multimodal/03_visual_question_answering/README.md',
]


def token_pattern(token: str) -> str:
    return rf'(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])'


def assert_has_track_reference(testcase: unittest.TestCase, text: str, token: str, rel: str) -> None:
    testcase.assertRegex(text, token_pattern(token), f'{rel} missing track reference {token}')


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
                assert_has_track_reference(self, text, token, rel)
            for token in RETIRED_TRACK_ROOTS:
                self.assertNotRegex(text, token_pattern(token), f'{rel} still mentions retired root {token}')

    def test_active_moved_curriculum_docs_do_not_reference_retired_roots(self) -> None:
        for rel in ACTIVE_MOVED_CURRICULUM_DOCS:
            text = (ROOT / rel).read_text(encoding='utf-8')
            for token in RETIRED_TRACK_ROOTS:
                self.assertNotRegex(text, token_pattern(token), f'{rel} still mentions retired root {token}')


if __name__ == '__main__':
    unittest.main()
