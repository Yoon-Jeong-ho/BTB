from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
VOCAB = {
    '<pad>': 0,
    '<bos>': 1,
    '<eos>': 2,
    '연구자는': 3,
    '긴': 4,
    '문맥을': 5,
    '천천히': 6,
    '읽는다': 7,
    '[MASK]': 8,
    '<extra_id_0>': 9,
    '<extra_id_1>': 10,
}
IGNORE_INDEX = -100
CONTEXT_WINDOW = 4


def _build_logits(target_ids: torch.Tensor, *, confusion_offset: int) -> torch.Tensor:
    vocab_size = len(VOCAB)
    logits = torch.full((target_ids.shape[0], vocab_size), -2.25, dtype=torch.float32)
    for index, target_id in enumerate(target_ids.tolist()):
        if target_id == IGNORE_INDEX:
            logits[index] = 0.0
            continue
        logits[index, target_id] = 2.4 + 0.05 * index
        logits[index, (target_id + confusion_offset) % vocab_size] = 1.15
        logits[index, (target_id + confusion_offset + 3) % vocab_size] = 0.45
    return logits


def _masked_mean_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    losses = F.cross_entropy(logits, targets, reduction='none', ignore_index=IGNORE_INDEX)
    kept = losses[targets != IGNORE_INDEX]
    return float(kept.mean().item())


def run() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(7)
    total_slots = 6

    causal_targets = torch.tensor(
        [
            VOCAB['연구자는'],
            VOCAB['긴'],
            VOCAB['문맥을'],
            VOCAB['천천히'],
            VOCAB['읽는다'],
            VOCAB['<eos>'],
        ]
    )
    masked_targets = torch.tensor(
        [
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            VOCAB['문맥을'],
            VOCAB['천천히'],
            IGNORE_INDEX,
            IGNORE_INDEX,
        ]
    )
    span_targets = torch.tensor(
        [
            VOCAB['<extra_id_0>'],
            VOCAB['문맥을'],
            VOCAB['천천히'],
            VOCAB['<extra_id_1>'],
        ]
    )

    causal_logits = _build_logits(causal_targets, confusion_offset=2)
    masked_logits = _build_logits(masked_targets, confusion_offset=4)
    span_logits = _build_logits(span_targets, confusion_offset=5)

    masked_scored_tokens = int((masked_targets != IGNORE_INDEX).sum().item())

    objectives = {
        'causal_lm': {
            'mean_loss': round(_masked_mean_loss(causal_logits, causal_targets), 6),
            'scored_tokens': int(causal_targets.numel()),
            'loss_mask_density': round(causal_targets.numel() / total_slots, 6),
            'target_length': int(causal_targets.numel()),
        },
        'masked_lm': {
            'mean_loss': round(_masked_mean_loss(masked_logits, masked_targets), 6),
            'scored_tokens': masked_scored_tokens,
            'loss_mask_density': round(masked_scored_tokens / total_slots, 6),
            'target_length': int(masked_targets.numel()),
        },
        'span_corruption': {
            'mean_loss': round(_masked_mean_loss(span_logits, span_targets), 6),
            'scored_tokens': int(span_targets.numel()),
            'loss_mask_density': round(span_targets.numel() / total_slots, 6),
            'decoder_target_length': int(span_targets.numel()),
            'encoder_input_length': 5,
        },
    }

    density_ranking = [
        name
        for name, _ in sorted(
            objectives.items(),
            key=lambda item: (-item[1]['loss_mask_density'], item[0]),
        )
    ]

    metrics = {
        'device': 'cpu',
        'seed': 7,
        'vocab_size': len(VOCAB),
        'context_window_tokens': CONTEXT_WINDOW,
        'objectives': objectives,
        'density_ranking': density_ranking,
        'context_window': {
            'causal_focus_visible_tokens': ['연구자는', '긴', '문맥을', '천천히'],
            'masked_focus_visible_tokens': ['긴', '[MASK]', '[MASK]', '읽는다'],
            'span_encoder_visible_tokens': ['연구자는', '긴', '<extra_id_0>', '읽는다'],
            'causal_future_blocked': True,
            'masked_middle_token_sees_both_sides': True,
            'span_decoder_reads_previous_targets_only': True,
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
