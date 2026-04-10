from __future__ import annotations

from typing import Any

VOCAB = [
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '자',
    '##연',
    '##어',
    '##를',
    '좋',
    '##아',
    '##해',
    '##요',
    '토',
    '##크',
    '##나',
    '##이',
    '##저',
    '##가',
    '필',
    '##합',
    '##니',
    '##다',
    '임',
    '##베',
    '##딩',
    '##은',
]
VOCAB_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
SENTENCES = [
    '자연어를 좋아해요',
    '토크나이저가 필요합니다',
    '임베딩은 초거대모델',
]
SPECIAL_TOKEN_IDS = {
    'pad': VOCAB_TO_ID['[PAD]'],
    'unk': VOCAB_TO_ID['[UNK]'],
    'cls': VOCAB_TO_ID['[CLS]'],
    'sep': VOCAB_TO_ID['[SEP]'],
}


def split_word_to_subwords(word: str) -> list[str]:
    if word in VOCAB_TO_ID:
        return [word]

    pieces: list[str] = []
    cursor = 0
    while cursor < len(word):
        matched = None
        match_end = cursor
        for end in range(len(word), cursor, -1):
            piece = word[cursor:end]
            candidate = piece if cursor == 0 else f'##{piece}'
            if candidate in VOCAB_TO_ID:
                matched = candidate
                match_end = end
                break

        if matched is None:
            return ['[UNK]']

        pieces.append(matched)
        cursor = match_end

    return pieces


def encode_sentence(sentence: str) -> dict[str, Any]:
    words = sentence.split()
    subwords: list[str] = []
    for word in words:
        subwords.extend(split_word_to_subwords(word))

    tokens = ['[CLS]', *subwords, '[SEP]']
    token_ids = [VOCAB_TO_ID.get(token, SPECIAL_TOKEN_IDS['unk']) for token in tokens]
    return {
        'sentence': sentence,
        'words': words,
        'subwords': subwords,
        'tokens': tokens,
        'token_ids': token_ids,
        'word_count': len(words),
        'subword_count': len(subwords),
        'sequence_length_with_special_tokens': len(tokens),
        'unknown_token_count': subwords.count('[UNK]'),
    }


def build_encoded_examples(sentences: list[str] | None = None) -> list[dict[str, Any]]:
    source = SENTENCES if sentences is None else sentences
    return [encode_sentence(sentence) for sentence in source]
