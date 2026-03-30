from __future__ import annotations

import json
from pathlib import Path

from tokenization_fixture import SPECIAL_TOKEN_IDS, VOCAB, build_encoded_examples

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'


def run() -> None:
    encoded_examples = build_encoded_examples()
    total_words = sum(int(example['word_count']) for example in encoded_examples)
    total_subwords = sum(int(example['subword_count']) for example in encoded_examples)
    max_sequence_length = max(
        (int(example['sequence_length_with_special_tokens']) for example in encoded_examples),
        default=0,
    )

    metrics = {
        'vocab_size': len(VOCAB),
        'special_token_ids': SPECIAL_TOKEN_IDS,
        'examples': encoded_examples,
        'total_word_count': total_words,
        'total_subword_count': total_subwords,
        'subword_expansion_ratio': round(total_subwords / total_words, 4) if total_words else 0.0,
        'unknown_token_total': sum(int(example['unknown_token_count']) for example in encoded_examples),
        'max_sequence_length_with_special_tokens': max_sequence_length,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
