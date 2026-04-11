from __future__ import annotations

import json
import math
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'

INITIAL_REWARDS = [0.42, 0.57, 0.38, 0.50]
VERIFIER_BONUSES = [0.16, 0.12, 0.18, 0.10]
JUDGE_BONUSES = [0.10, 0.08, 0.11, 0.09]
KL_COSTS = [0.035, 0.042, 0.038, 0.045]
KL_GUARDRAIL = 0.12


def _round(value: float) -> float:
    return round(value, 6)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _training_trace() -> list[dict[str, float | int]]:
    trace = []
    for step in range(5):
        progress = step / 4
        reward_mean = _mean(INITIAL_REWARDS) + progress * 0.18
        advantage_mean = 0.015 + progress * 0.19
        kl = 0.02 + progress * 0.065
        answer_accuracy = 0.50 + progress * 0.18
        verifier_consistency = 0.54 + progress * 0.20
        judge_win_rate = 0.55 + progress * 0.17
        trace.append(
            {
                'step': step,
                'reward_mean': _round(reward_mean),
                'advantage_mean': _round(advantage_mean),
                'kl': _round(kl),
                'answer_accuracy': _round(answer_accuracy),
                'verifier_consistency': _round(verifier_consistency),
                'judge_win_rate': _round(judge_win_rate),
            }
        )
    return trace


def run() -> None:
    shaped_rewards = [
        reward + verifier + judge - kl
        for reward, verifier, judge, kl in zip(INITIAL_REWARDS, VERIFIER_BONUSES, JUDGE_BONUSES, KL_COSTS)
    ]
    centered_advantages = [reward - _mean(INITIAL_REWARDS) for reward in INITIAL_REWARDS]
    trace = _training_trace()
    first = trace[0]
    last = trace[-1]

    metrics = {
        'device': 'cpu',
        'simulation': 'tiny_numeric_reasoning_rl',
        'seed': 0,
        'rollout_batch_size': len(INITIAL_REWARDS),
        'tensor_shapes': {
            'reward_scores': [len(INITIAL_REWARDS)],
            'advantages': [len(INITIAL_REWARDS)],
            'verifier_scores': [len(VERIFIER_BONUSES)],
            'judge_scores': [len(JUDGE_BONUSES)],
        },
        'reward_components': {
            'initial_rewards': [_round(value) for value in INITIAL_REWARDS],
            'verifier_bonuses': [_round(value) for value in VERIFIER_BONUSES],
            'judge_bonuses': [_round(value) for value in JUDGE_BONUSES],
            'kl_costs': [_round(value) for value in KL_COSTS],
            'shaped_rewards': [_round(value) for value in shaped_rewards],
            'initial_reward_std': _round(_std(INITIAL_REWARDS)),
        },
        'policy_update': {
            'update_family': 'PPO-family clipped advantage sketch',
            'reward_mean_before': _round(float(first['reward_mean'])),
            'reward_mean_after': _round(float(last['reward_mean'])),
            'advantage_mean_before': _round(float(first['advantage_mean'])),
            'advantage_mean_after': _round(float(last['advantage_mean'])),
            'centered_advantages': [_round(value) for value in centered_advantages],
            'policy_loss_proxy': _round(-float(last['advantage_mean']) + 0.4 * float(last['kl'])),
            'kl_after': _round(float(last['kl'])),
            'kl_guardrail': KL_GUARDRAIL,
            'kl_anchor_enabled': True,
            'no_gpu_required': True,
        },
        'reasoning_eval': {
            'answer_accuracy_before': _round(float(first['answer_accuracy'])),
            'answer_accuracy_after': _round(float(last['answer_accuracy'])),
            'verifier_consistency_before': _round(float(first['verifier_consistency'])),
            'verifier_consistency_after': _round(float(last['verifier_consistency'])),
            'judge_win_rate_after': _round(float(last['judge_win_rate'])),
            'judge_length_bias_flag': True,
            'process_signal_note': 'verifier consistency improves, but judge length bias remains a separate regression slice.',
        },
        'failure_mode_probes': {
            'highest_risk': 'reward_hacking',
            'verbosity_delta': 0.09,
            'over_refusal_delta': 0.05,
            'style_bias_delta': 0.04,
            'format_gaming_delta': 0.03,
            'mitigation': 'held-out factuality/safety prompts plus verifier/judge disagreement review',
        },
        'training_trace': trace,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'device': metrics['device'],
        'simulation': metrics['simulation'],
        'policy_update': metrics['policy_update'],
        'reasoning_eval': metrics['reasoning_eval'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
