from __future__ import annotations

import json
from pathlib import Path

import torch

from tokenization_fixture import SPECIAL_TOKEN_IDS, VOCAB, build_encoded_examples

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
PAD_ID = SPECIAL_TOKEN_IDS['pad']


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)

    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [pad_id] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long)


def run() -> None:
    torch.manual_seed(7)

    encoded_examples = build_encoded_examples()
    sentence_token_ids = [list(example['token_ids']) for example in encoded_examples]
    input_ids = _pad_sequences(sentence_token_ids, PAD_ID)
    padding_mask = input_ids.ne(PAD_ID)

    embedding = torch.nn.Embedding(len(VOCAB), 6, padding_idx=PAD_ID)
    embedded = embedding(input_ids)
    masked_embeddings = embedded * padding_mask.unsqueeze(-1)
    pooled = masked_embeddings.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).clamp_min(1)

    pad_positions = input_ids.eq(PAD_ID)
    if pad_positions.any():
        pad_vectors = embedded[pad_positions]
        pad_vector_abs_max = round(float(pad_vectors.abs().max().item()), 6)
    else:
        pad_vector_abs_max = 0.0

    metrics = {
        'batch_size': int(input_ids.shape[0]),
        'sentence_order': [str(example['sentence']) for example in encoded_examples],
        'sequence_lengths_with_special_tokens': [
            int(example['sequence_length_with_special_tokens']) for example in encoded_examples
        ],
        'max_sequence_length': int(input_ids.shape[1]) if input_ids.ndim == 2 else 0,
        'embedding_dim': int(embedded.shape[-1]) if embedded.ndim == 3 else 0,
        'input_ids_shape': list(input_ids.shape),
        'embedding_weight_shape': list(embedding.weight.shape),
        'embedded_shape': list(embedded.shape),
        'padding_mask_shape': list(padding_mask.shape),
        'masked_embeddings_shape': list(masked_embeddings.shape),
        'pooled_shape': list(pooled.shape),
        'non_pad_counts': [int(count) for count in padding_mask.sum(dim=1).tolist()],
        'pad_token_row_is_zero': bool(
            torch.allclose(embedding.weight[PAD_ID], torch.zeros_like(embedding.weight[PAD_ID]))
        ),
        'pad_vector_abs_max': pad_vector_abs_max,
        'first_sentence_first_token_preview': [
            round(float(value), 6) for value in embedded[0, 0, :3].detach()
        ] if embedded.numel() else [],
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
