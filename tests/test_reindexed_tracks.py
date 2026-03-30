from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestReindexedTracks(unittest.TestCase):
    def test_new_track_dirs_exist(self) -> None:
        self.assertTrue((ROOT / '03_nlp').exists())
        self.assertTrue((ROOT / '05_multimodal').exists())

    def test_old_track_dirs_are_gone(self) -> None:
        self.assertFalse((ROOT / '02_nlp').exists())
        self.assertFalse((ROOT / '03_multimodal').exists())

    def test_stage_readmes_survive_move(self) -> None:
        nlp_text = (ROOT / '03_nlp' / 'README.md').read_text(encoding='utf-8')
        self.assertIn('텍스트 전처리 -> bag-of-words baseline -> pretrained LM finetuning -> error analysis', nlp_text)
        multimodal_text = (ROOT / '05_multimodal' / 'README.md').read_text(encoding='utf-8')
        self.assertIn('이미지와 텍스트를 같은 표현 공간에서 다루는 법', multimodal_text)

        for rel in [
            '03_nlp/01_text_classification/README.md',
            '03_nlp/02_named_entity_recognition/README.md',
            '03_nlp/03_machine_reading_comprehension/README.md',
            '05_multimodal/01_image_text_retrieval/README.md',
            '05_multimodal/02_image_captioning/README.md',
            '05_multimodal/03_visual_question_answering/README.md',
        ]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_root_docs_use_reindexed_paths(self) -> None:
        for rel in ['README.md', 'docs/00_program_map.md', 'scripts/README.md']:
            text = (ROOT / rel).read_text(encoding='utf-8')
            self.assertIn('03_nlp', text, rel)
            self.assertIn('05_multimodal', text, rel)
            self.assertNotIn('02_nlp/', text, rel)
            self.assertNotIn('03_multimodal/', text, rel)


if __name__ == '__main__':
    unittest.main()
