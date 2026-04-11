from __future__ import annotations

import json
import math
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
BETA = 0.8
REFERENCE_MARGINS = [0.00, 0.03, 0.08, -0.04]
INITIAL_POLICY_MARGINS = [-0.08, 0.05, 0.12, -0.03]
TARGET_INCREMENTS = [0.20, 0.14, 0.10, 0.18]
LABEL_LOGPROBS = [
    ('desirable', -1.86),
    ('desirable', -1.92),
    ('undesirable', -2.28),
    ('undesirable', -2.44),
]


def _round(value: float) -> float:
    return round(value, 6)


def _logistic_loss(value: float) -> float:
    return math.log1p(math.exp(-value))


def _pair_accuracy(margins: list[float]) -> float:
    return sum(1 for margin in margins if margin > 0) / len(margins)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _simulate_margin_updates() -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    for step in range(5):
        progress = step / 4
        margins = [
            before + progress * increment
            for before, increment in zip(INITIAL_POLICY_MARGINS, TARGET_INCREMENTS)
        ]
        dpo_advantages = [margin - ref for margin, ref in zip(margins, REFERENCE_MARGINS)]
        history.append(
            {
                'step': step,
                'avg_policy_margin': _round(_mean(margins)),
                'pair_accuracy': _round(_pair_accuracy(margins)),
                'avg_dpo_advantage': _round(_mean(dpo_advantages)),
                'reference_drift': _round(_mean([abs(margin - ref) for margin, ref in zip(margins, REFERENCE_MARGINS)])),
            }
        )
    return history


def _objective_losses(margins: list[float]) -> dict[str, float]:
    dpo = _mean([
        _logistic_loss(BETA * (margin - ref))
        for margin, ref in zip(margins, REFERENCE_MARGINS)
    ])
    orpo = _mean([0.18 + _logistic_loss(margin) for margin in margins])
    kto_terms: list[float] = []
    for label, logprob in LABEL_LOGPROBS:
        signed = 1.0 if label == 'desirable' else -1.0
        kto_terms.append(_logistic_loss(signed * (logprob + 2.15)))
    return {
        'dpo': _round(dpo),
        'orpo': _round(orpo),
        'kto': _round(_mean(kto_terms)),
    }


def run() -> None:
    history = _simulate_margin_updates()
    final_margins = [
        before + increment
        for before, increment in zip(INITIAL_POLICY_MARGINS, TARGET_INCREMENTS)
    ]
    final_drift = _mean([abs(margin - ref) for margin, ref in zip(final_margins, REFERENCE_MARGINS)])

    metrics = {
        'device': 'cpu',
        'simulation': 'tiny_numeric_policy',
        'seed': 0,
        'preference_pairs': 4,
        'reference_margins': [_round(value) for value in REFERENCE_MARGINS],
        'initial_policy_margins': [_round(value) for value in INITIAL_POLICY_MARGINS],
        'final_policy_margins': [_round(value) for value in final_margins],
        'history': history,
        'objective_losses': _objective_losses(final_margins),
        'policy_update': {
            'avg_margin_before': _round(_mean(INITIAL_POLICY_MARGINS)),
            'avg_margin_after': _round(_mean(final_margins)),
            'pair_accuracy_before': _round(_pair_accuracy(INITIAL_POLICY_MARGINS)),
            'pair_accuracy_after': _round(_pair_accuracy(final_margins)),
            'reference_drift_after': _round(final_drift),
            'reference_drift_guardrail': 0.2,
            'without_full_rl_loop': True,
            'update_note': 'offline chosen/rejected log-prob margin을 직접 키우는 tiny numeric simulation이다.',
        },
        'contrast': {
            'pairwise_reference_method': 'DPO',
            'pairwise_no_reference_anchor': 'ORPO',
            'label_only_method': 'KTO',
            'data_requirement_note': 'DPO/ORPO는 strict pair가 자연스럽고, KTO는 desirable/undesirable label만으로도 toy update를 만들 수 있다.',
        },
        'eval_tradeoffs': {
            'helpfulness_gain_proxy': 0.18,
            'format_following_gain_proxy': 0.14,
            'refusal_overreach_delta': 0.04,
            'verbosity_delta': 0.06,
            'tradeoff_note': 'alignment eval은 win rate와 함께 factuality, refusal balance, verbosity를 분리해야 한다.',
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
