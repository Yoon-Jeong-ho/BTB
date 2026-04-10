from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
SEED = 13
ADAPT_STEPS = 20
GENERAL_GUARDRAIL = 0.12
VOCAB = [
    '<bos>',
    '<eos>',
    '회의',
    '문서',
    '일정',
    '공유',
    '검토',
    '환자',
    '혈압',
    '투약',
    '진단',
    '처방',
    '기록',
    '조정',
]
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
GENERAL_TRAIN = [
    ['<bos>', '회의', '일정', '공유', '<eos>'],
    ['<bos>', '문서', '검토', '공유', '<eos>'],
    ['<bos>', '회의', '문서', '검토', '<eos>'],
    ['<bos>', '일정', '공유', '검토', '<eos>'],
]
GENERAL_VAL = [
    ['<bos>', '회의', '검토', '공유', '<eos>'],
    ['<bos>', '문서', '일정', '공유', '<eos>'],
]
DOMAIN_TRAIN = [
    ['<bos>', '환자', '혈압', '기록', '<eos>'],
    ['<bos>', '환자', '투약', '조정', '<eos>'],
    ['<bos>', '진단', '처방', '기록', '<eos>'],
    ['<bos>', '혈압', '진단', '처방', '<eos>'],
]
DOMAIN_VAL = [
    ['<bos>', '환자', '진단', '처방', '<eos>'],
    ['<bos>', '혈압', '투약', '기록', '<eos>'],
]


class TinyBigramLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(inputs))


def _pairs(sequences: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[int] = []
    targets: list[int] = []
    for sequence in sequences:
        ids = [TOKEN_TO_ID[token] for token in sequence]
        inputs.extend(ids[:-1])
        targets.extend(ids[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _loss(model: TinyBigramLM, sequences: list[list[str]]) -> float:
    inputs, targets = _pairs(sequences)
    with torch.no_grad():
        return round(float(F.cross_entropy(model(inputs), targets).item()), 6)


def _train_epoch(model: TinyBigramLM, sequences: list[list[str]], *, lr: float) -> None:
    inputs, targets = _pairs(sequences)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()


def _train_base() -> TinyBigramLM:
    torch.manual_seed(SEED)
    model = TinyBigramLM(len(VOCAB))
    for _ in range(70):
        _train_epoch(model, GENERAL_TRAIN, lr=0.22)
    return model


def _recommended_stop(history: list[dict[str, float | int]], base_general: float) -> int:
    candidates = [
        (float(point['domain_loss']), int(point['step']))
        for point in history
        if round(float(point['general_loss']) - base_general, 6) <= GENERAL_GUARDRAIL
    ]
    if not candidates:
        return 0
    return min(candidates)[1]


def _guardrail_exceeded(history: list[dict[str, float | int]], base_general: float) -> int:
    for point in history:
        if round(float(point['general_loss']) - base_general, 6) > GENERAL_GUARDRAIL:
            return int(point['step'])
    return 0


def _adapt(
    base_model: TinyBigramLM,
    *,
    schedule: list[list[list[str]]],
    domain_share: float,
    general_share: float,
    lr: float,
    base_domain: float,
    base_general: float,
) -> dict[str, object]:
    model = TinyBigramLM(len(VOCAB))
    model.load_state_dict(deepcopy(base_model.state_dict()))
    history: list[dict[str, float | int]] = [
        {
            'step': 0,
            'domain_loss': base_domain,
            'general_loss': base_general,
        }
    ]
    for step, sequences in enumerate(schedule, start=1):
        _train_epoch(model, sequences, lr=lr)
        history.append(
            {
                'step': step,
                'domain_loss': _loss(model, DOMAIN_VAL),
                'general_loss': _loss(model, GENERAL_VAL),
            }
        )
    final = history[-1]
    recommended = _recommended_stop(history, base_general)
    exceeded = _guardrail_exceeded(history, base_general)
    return {
        'domain_share': domain_share,
        'general_share': general_share,
        'history': history,
        'final_domain_loss': final['domain_loss'],
        'final_general_loss': final['general_loss'],
        'in_domain_gain': round(base_domain - float(final['domain_loss']), 6),
        'general_regression': round(float(final['general_loss']) - base_general, 6),
        'recommended_stop_step': recommended,
        'guardrail_exceeded_step': exceeded,
    }


def _selection_profile() -> dict[str, object]:
    curated = {
        'document_count': 4,
        'duplicate_rate': 0.0,
        'target_distribution_match': 0.91,
        'contamination_risk': 0.04,
    }
    noisy = {
        'document_count': 10,
        'duplicate_rate': 0.45,
        'target_distribution_match': 0.58,
        'contamination_risk': 0.2,
    }
    for profile in (curated, noisy):
        profile['selection_score'] = round(
            profile['target_distribution_match'] * (1.0 - profile['duplicate_rate']) * (1.0 - profile['contamination_risk']),
            6,
        )
    return {
        'curated_domain': curated,
        'noisy_domain': noisy,
        'preferred': 'curated_domain',
        'scoring_note': '목표 분포 적합도에서 중복률과 contamination risk를 할인한 toy 점수다.',
    }


def run() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    base_model = _train_base()
    base_general = _loss(base_model, GENERAL_VAL)
    base_domain = _loss(base_model, DOMAIN_VAL)

    pure_schedule = [DOMAIN_TRAIN for _ in range(ADAPT_STEPS)]
    replay_schedule = [DOMAIN_TRAIN + DOMAIN_TRAIN + GENERAL_TRAIN for _ in range(ADAPT_STEPS)]

    strategies = {
        'pure_domain': _adapt(
            base_model,
            schedule=pure_schedule,
            domain_share=1.0,
            general_share=0.0,
            lr=0.25,
            base_domain=base_domain,
            base_general=base_general,
        ),
        'replay_mixture': _adapt(
            base_model,
            schedule=replay_schedule,
            domain_share=0.67,
            general_share=0.33,
            lr=0.25,
            base_domain=base_domain,
            base_general=base_general,
        ),
    }

    metrics = {
        'device': 'cpu',
        'seed': SEED,
        'vocab_size': len(VOCAB),
        'objective_kept_constant': 'causal_lm_bigram_next_token',
        'base_losses': {
            'general': base_general,
            'domain': base_domain,
        },
        'stopping_guardrail': {
            'max_general_regression': GENERAL_GUARDRAIL,
            'rule': 'domain loss가 가장 낮으면서 general regression이 guardrail 이하인 step을 추천한다.',
        },
        'strategies': strategies,
        'data_selection': _selection_profile(),
        'comparison': {
            'pure_domain_adapts_faster': strategies['pure_domain']['in_domain_gain'] > strategies['replay_mixture']['in_domain_gain'],
            'replay_preserves_general_better': strategies['replay_mixture']['general_regression'] < strategies['pure_domain']['general_regression'],
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
