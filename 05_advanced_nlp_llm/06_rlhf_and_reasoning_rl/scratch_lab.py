from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
OUT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = OUT_DIR / 'metrics.json'
FIGURE_PATH = OUT_DIR / 'rlhf_reasoning_reward.svg'

ROLL_OUTS = [
    {
        'id': 'math_check',
        'prompt': '두 자리 덧셈을 단계적으로 풀고 검산하라.',
        'chosen': {
            'response': '37+48=85를 계산하고 48+37로 다시 검산한다.',
            'verifier_score': 0.92,
            'judge_score': 0.84,
            'format_score': 0.78,
            'safety_score': 0.80,
            'reasoning_steps': 3,
        },
        'rejected': {
            'response': '37+48은 대략 90이므로 답은 90이다.',
            'verifier_score': 0.24,
            'judge_score': 0.34,
            'format_score': 0.55,
            'safety_score': 0.80,
            'reasoning_steps': 1,
        },
    },
    {
        'id': 'grounded_refusal',
        'prompt': '확실하지 않은 의료 질문에는 근거와 한계를 분리하라.',
        'chosen': {
            'response': '증상만으로 진단하지 않고 위험 신호와 전문 상담 기준을 안내한다.',
            'verifier_score': 0.88,
            'judge_score': 0.81,
            'format_score': 0.74,
            'safety_score': 0.95,
            'reasoning_steps': 2,
        },
        'rejected': {
            'response': '가능성이 높은 질병 하나를 확정적으로 말한다.',
            'verifier_score': 0.18,
            'judge_score': 0.30,
            'format_score': 0.52,
            'safety_score': 0.18,
            'reasoning_steps': 1,
        },
    },
    {
        'id': 'tool_consistency',
        'prompt': '표의 숫자 합계를 근거로 결론을 말하라.',
        'chosen': {
            'response': '각 행 합계를 다시 계산하고 최종 합계와 결론을 연결한다.',
            'verifier_score': 0.86,
            'judge_score': 0.79,
            'format_score': 0.82,
            'safety_score': 0.82,
            'reasoning_steps': 4,
        },
        'rejected': {
            'response': '표를 자세히 설명하지만 합계는 한 번만 추정한다.',
            'verifier_score': 0.42,
            'judge_score': 0.58,
            'format_score': 0.76,
            'safety_score': 0.82,
            'reasoning_steps': 5,
        },
    },
    {
        'id': 'concise_reasoning',
        'prompt': '짧은 근거와 정답만 요구하는 문제에 답하라.',
        'chosen': {
            'response': '필요한 한 줄 근거와 최종 답만 제시한다.',
            'verifier_score': 0.80,
            'judge_score': 0.77,
            'format_score': 0.90,
            'safety_score': 0.82,
            'reasoning_steps': 2,
        },
        'rejected': {
            'response': '여러 가정과 긴 추론 흔적을 덧붙여 답한다.',
            'verifier_score': 0.70,
            'judge_score': 0.64,
            'format_score': 0.38,
            'safety_score': 0.82,
            'reasoning_steps': 8,
        },
    },
]

WEIGHTS = {
    'verifier_score': 0.42,
    'judge_score': 0.32,
    'format_score': 0.14,
    'safety_score': 0.12,
}
PROCESS_REWARD_WEIGHT = 0.35
KL_ANCHOR = 0.07


def _round(value: float) -> float:
    return round(value, 6)


def _reward(candidate: dict[str, object]) -> float:
    weighted = sum(float(candidate[key]) * weight for key, weight in WEIGHTS.items())
    step_count = int(candidate['reasoning_steps'])
    concise_process_bonus = 0.08 if 2 <= step_count <= 4 else -0.05
    return _round(weighted + PROCESS_REWARD_WEIGHT * concise_process_bonus - KL_ANCHOR)


def _candidate_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in ROLL_OUTS:
        chosen_reward = _reward(item['chosen'])
        rejected_reward = _reward(item['rejected'])
        rows.append(
            {
                'id': item['id'],
                'prompt': item['prompt'],
                'chosen_reward': chosen_reward,
                'rejected_reward': rejected_reward,
                'reward_margin': _round(chosen_reward - rejected_reward),
                'chosen_verifier': item['chosen']['verifier_score'],
                'rejected_verifier': item['rejected']['verifier_score'],
                'chosen_reasoning_steps': item['chosen']['reasoning_steps'],
                'rejected_reasoning_steps': item['rejected']['reasoning_steps'],
                'judge_prefers_chosen': float(item['chosen']['judge_score']) > float(item['rejected']['judge_score']),
                'verifier_prefers_chosen': float(item['chosen']['verifier_score']) > float(item['rejected']['verifier_score']),
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _render_svg(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width = 790
    height = 430
    left = 86
    top = 92
    chart_width = 570
    chart_height = 210
    max_value = 1.05
    row_gap = chart_width / len(rows)
    bar_width = 34

    def y_for(value: float) -> float:
        return top + chart_height - (value / max_value) * chart_height

    grid: list[str] = []
    for tick in range(6):
        value = tick * 0.2
        y = y_for(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#dce6f3" />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#4b5f78">{value:.1f}</text>')

    bars: list[str] = []
    for index, row in enumerate(rows):
        center = left + row_gap * index + row_gap / 2
        for offset, key, color in [
            (-bar_width, 'chosen_reward', '#4f7cff'),
            (0, 'rejected_reward', '#f2994a'),
            (bar_width, 'chosen_verifier', '#28a36a'),
        ]:
            value = float(row[key])
            y = y_for(value)
            x = center + offset - bar_width / 2
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{top + chart_height - y:.1f}" fill="{color}" rx="5" />')
            bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle" fill="#263a52">{value:.2f}</text>')
        bars.append(f'<text x="{center:.1f}" y="{top + chart_height + 24}" font-size="11" text-anchor="middle" fill="#263a52">{escape(str(row["id"]).split("_")[0])}</text>')

    legend = (
        '<rect x="82" y="342" width="16" height="16" fill="#4f7cff" rx="3" />'
        '<text x="106" y="355" font-size="13" fill="#263a52">chosen reward</text>'
        '<rect x="230" y="342" width="16" height="16" fill="#f2994a" rx="3" />'
        '<text x="254" y="355" font-size="13" fill="#263a52">rejected reward</text>'
        '<rect x="390" y="342" width="16" height="16" fill="#28a36a" rx="3" />'
        '<text x="414" y="355" font-size="13" fill="#263a52">verifier bonus</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="36" font-size="24" font-weight="bold" fill="#10203a">RLHF and reasoning RL reward signals</text>'
        '<text x="28" y="62" font-size="14" fill="#42556d">Toy reward model + verifier bonus before a PPO-family policy update</text>'
        '<text x="28" y="84" font-size="15" font-weight="bold" fill="#10203a">reward / verifier score</text>'
        + ''.join(grid)
        + ''.join(bars)
        + legend
        + '<text x="82" y="392" font-size="13" fill="#42556d">Reward is a preference proxy: useful for policy shaping, unsafe as a truth engine.</text>'
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    rows = _candidate_metrics()
    chosen_rewards = [float(row['chosen_reward']) for row in rows]
    rejected_rewards = [float(row['rejected_reward']) for row in rows]
    verifier_wins = sum(1 for row in rows if row['verifier_prefers_chosen']) / len(rows)
    judge_wins = sum(1 for row in rows if row['judge_prefers_chosen']) / len(rows)
    avg_chosen_steps = _mean([float(row['chosen_reasoning_steps']) for row in rows])
    avg_rejected_steps = _mean([float(row['rejected_reasoning_steps']) for row in rows])
    _render_svg(rows)

    metrics = {
        'setup': {
            'unit': '06_rlhf_and_reasoning_rl',
            'mode': 'cpu_safe_deterministic_toy_rlhf_reasoning_rl',
            'cpu_safe': True,
            'seed': 0,
            'no_real_llm_training': True,
        },
        'reward_model_batch': {
            'prompt_count': len(ROLL_OUTS),
            'candidate_count': len(ROLL_OUTS) * 2,
            'chosen_rejected_pairs': len(ROLL_OUTS),
            'avg_reward_chosen': _round(_mean(chosen_rewards)),
            'avg_reward_rejected': _round(_mean(rejected_rewards)),
            'avg_reward_margin': _round(_mean([c - r for c, r in zip(chosen_rewards, rejected_rewards)])),
            'reward_model_intuition': 'preference proxy, not truth engine',
            'rubric_weights': WEIGHTS,
        },
        'candidate_metrics': rows,
        'rlhf_loop_view': {
            'steps': ['sample_prompts', 'policy_rollouts', 'score_rewards', 'ppo_family_update', 'regression_eval'],
            'reward_source': 'toy reward model + verifier bonus + judge preference signal',
            'policy_update_style': 'PPO-family clipped advantage sketch, not full training',
            'kl_anchor_enabled': True,
            'kl_anchor_penalty': KL_ANCHOR,
            'regression_watch': ['reward_hacking', 'verbosity', 'over-refusal', 'format drift', 'style bias'],
        },
        'reasoning_signal': {
            'outcome_reward_weight': 0.65,
            'process_reward_weight': PROCESS_REWARD_WEIGHT,
            'verifier_pass_rate': _round(verifier_wins),
            'judge_preference_win_rate': _round(judge_wins),
            'avg_chosen_reasoning_steps': _round(avg_chosen_steps),
            'avg_rejected_reasoning_steps': _round(avg_rejected_steps),
            'longer_trace_is_always_better': False,
            'reward_shaping_note': '좋은 reasoning RL은 긴 trace보다 검증 가능성, 수정 능력, 최종 답 정확도를 함께 보상한다.',
        },
        'failure_modes': {
            'primary_watch': 'reward_hacking',
            'length_bias_flag': True,
            'verbosity_risk': _round(max(0.0, avg_rejected_steps - avg_chosen_steps) / 10.0),
            'over_refusal_risk': 0.11,
            'style_bias_risk': 0.14,
            'note': 'judge와 verifier 모두 gaming될 수 있으므로 held-out factuality/safety slices가 필요하다.',
        },
        'figure_path': 'artifacts/scratch-manual/rlhf_reasoning_reward.svg',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'reward_model_batch': metrics['reward_model_batch'],
        'rlhf_loop_view': metrics['rlhf_loop_view'],
        'reasoning_signal': metrics['reasoning_signal'],
        'figure_path': metrics['figure_path'],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
