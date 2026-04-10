from __future__ import annotations

import json
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
PAD_ID = 0
VOCAB = {
    '[PAD]': 0,
    '[BOS]': 1,
    '학생은': 2,
    '어제': 3,
    '도서관에서': 4,
    '책을': 5,
    '읽었다': 6,
    '요약': 7,
}
ENCODER_INPUT_IDS = torch.tensor(
    [
        [2, 3, 4, 5, 6],
        [2, 4, 5, 6, 0],
    ],
    dtype=torch.long,
)
DECODER_INPUT_IDS = torch.tensor(
    [
        [1, 5, 6, 7, 0],
        [1, 4, 6, 7, 0],
    ],
    dtype=torch.long,
)
EMBED_DIM = 8
NUM_HEADS = 2


def _future_attention_stats(attention_weights: torch.Tensor) -> tuple[float, float]:
    seq_len = int(attention_weights.shape[-1])
    future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    selected = attention_weights.masked_select(future_mask[None, None, :, :].expand_as(attention_weights))
    if selected.numel() == 0:
        return 0.0, 0.0
    return float(selected.abs().max().item()), float(selected.mean().item())


def run() -> None:
    torch.manual_seed(23)

    embedding = torch.nn.Embedding(len(VOCAB), EMBED_DIM, padding_idx=PAD_ID)
    encoder_states = embedding(ENCODER_INPUT_IDS)
    decoder_states = embedding(DECODER_INPUT_IDS)

    encoder_padding_mask = ENCODER_INPUT_IDS.eq(PAD_ID)
    decoder_padding_mask = DECODER_INPUT_IDS.eq(PAD_ID)
    seq_len = int(ENCODER_INPUT_IDS.shape[1])
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    encoder_attention = torch.nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
        batch_first=True,
    )
    decoder_attention = torch.nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
        batch_first=True,
    )
    cross_attention = torch.nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
        batch_first=True,
    )

    encoder_hidden, encoder_weights = encoder_attention(
        encoder_states,
        encoder_states,
        encoder_states,
        key_padding_mask=encoder_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    decoder_hidden, decoder_weights = decoder_attention(
        decoder_states,
        decoder_states,
        decoder_states,
        key_padding_mask=decoder_padding_mask,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    cross_hidden, cross_weights = cross_attention(
        decoder_hidden,
        encoder_hidden,
        encoder_hidden,
        key_padding_mask=encoder_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    decoder_norm = torch.nn.LayerNorm(EMBED_DIM)(decoder_states + decoder_hidden)
    cross_norm = torch.nn.LayerNorm(EMBED_DIM)(decoder_norm + cross_hidden)
    feed_forward = torch.nn.Sequential(
        torch.nn.Linear(EMBED_DIM, EMBED_DIM * 2),
        torch.nn.GELU(),
        torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
    )
    decoder_block_output = torch.nn.LayerNorm(EMBED_DIM)(cross_norm + feed_forward(cross_norm))

    decoder_future_attention_max, _ = _future_attention_stats(decoder_weights)
    _, encoder_future_attention_mean = _future_attention_stats(encoder_weights)
    per_head_difference_mean = float(
        (encoder_weights[0, 0] - encoder_weights[0, 1]).abs().mean().item()
    )

    metrics = {
        'device': str(encoder_hidden.device),
        'batch_size': int(ENCODER_INPUT_IDS.shape[0]),
        'sequence_length': seq_len,
        'embed_dim': EMBED_DIM,
        'num_heads': NUM_HEADS,
        'encoder_hidden_shape': list(encoder_hidden.shape),
        'decoder_hidden_shape': list(decoder_hidden.shape),
        'cross_attention_output_shape': list(cross_hidden.shape),
        'decoder_block_output_shape': list(decoder_block_output.shape),
        'encoder_attention_weights_shape': list(encoder_weights.shape),
        'decoder_attention_weights_shape': list(decoder_weights.shape),
        'cross_attention_weights_shape': list(cross_weights.shape),
        'cross_attention_used': True,
        'decoder_future_attention_max': round(decoder_future_attention_max, 8),
        'encoder_future_attention_mean': round(encoder_future_attention_mean, 8),
        'per_head_difference_mean': round(per_head_difference_mean, 8),
        'encoder_valid_token_counts': [int(count) for count in (~encoder_padding_mask).sum(dim=1).tolist()],
        'decoder_valid_token_counts': [int(count) for count in (~decoder_padding_mask).sum(dim=1).tolist()],
        'first_decoder_query_cross_weights': [
            round(float(value), 6) for value in cross_weights[0, 0, 0].detach().tolist()
        ],
        'recurrent_relief': {
            'recurrent_steps': seq_len,
            'attention_parallel_rounds': 1,
            'longest_dependency_path_rnn': seq_len - 1,
            'longest_dependency_path_attention': 1,
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
