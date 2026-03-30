from __future__ import annotations

import json
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
TOKEN_TO_ID = {
    '[PAD]': 0,
    '[CLS]': 1,
    '나는': 2,
    '커피를': 3,
    '좋아해요': 4,
    '차를': 5,
    '마셔요': 6,
    '정말': 7,
}
INPUT_IDS = torch.tensor(
    [
        [1, 2, 3, 4, 0],
        [1, 2, 7, 4, 0],
        [1, 2, 5, 6, 4],
    ],
    dtype=torch.long,
)
PAD_ID = TOKEN_TO_ID['[PAD]']
EMBED_DIM = 8
NUM_HEADS = 2


def run() -> None:
    torch.manual_seed(11)

    embedding = torch.nn.Embedding(len(TOKEN_TO_ID), EMBED_DIM, padding_idx=PAD_ID)
    hidden_states = embedding(INPUT_IDS)
    key_padding_mask = INPUT_IDS.eq(PAD_ID)
    seq_len = int(INPUT_IDS.shape[1])
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    attention = torch.nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
        batch_first=True,
    )
    attention_output, attention_weights = attention(
        hidden_states,
        hidden_states,
        hidden_states,
        key_padding_mask=key_padding_mask,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    residual_after_attention = hidden_states + attention_output
    norm1 = torch.nn.LayerNorm(EMBED_DIM)
    normed_after_attention = norm1(residual_after_attention)
    feed_forward = torch.nn.Sequential(
        torch.nn.Linear(EMBED_DIM, EMBED_DIM * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
    )
    ff_output = feed_forward(normed_after_attention)
    transformer_block_output = torch.nn.LayerNorm(EMBED_DIM)(normed_after_attention + ff_output)

    pad_key_attention_max = 0.0
    if key_padding_mask.any():
        masked_columns = attention_weights.masked_select(
            key_padding_mask[:, None, None, :].expand_as(attention_weights)
        )
        if masked_columns.numel() > 0:
            pad_key_attention_max = round(float(masked_columns.abs().max().item()), 6)

    future_attention_max = 0.0
    future_only = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1,
    )
    future_values = attention_weights.masked_select(future_only[None, None, :, :].expand_as(attention_weights))
    if future_values.numel() > 0:
        future_attention_max = round(float(future_values.abs().max().item()), 6)

    metrics = {
        'token_to_id': TOKEN_TO_ID,
        'batch_size': int(INPUT_IDS.shape[0]),
        'sequence_length': seq_len,
        'embedding_dim': EMBED_DIM,
        'num_heads': NUM_HEADS,
        'input_ids_shape': list(INPUT_IDS.shape),
        'embedded_shape': list(hidden_states.shape),
        'key_padding_mask_shape': list(key_padding_mask.shape),
        'causal_mask_shape': list(causal_mask.shape),
        'attention_output_shape': list(attention_output.shape),
        'attention_weights_shape': list(attention_weights.shape),
        'transformer_block_output_shape': list(transformer_block_output.shape),
        'valid_token_counts': [int(count) for count in (~key_padding_mask).sum(dim=1).tolist()],
        'pad_token_row_is_zero': bool(
            torch.allclose(embedding.weight[PAD_ID], torch.zeros_like(embedding.weight[PAD_ID]))
        ),
        'pad_key_attention_max': pad_key_attention_max,
        'future_attention_max': future_attention_max,
        'first_head_last_query_weights_batch0': [
            round(float(value), 6) for value in attention_weights[0, 0, -1].detach().tolist()
        ],
        'first_token_block_preview': [
            round(float(value), 6) for value in transformer_block_output[0, 0, :4].detach().tolist()
        ],
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
