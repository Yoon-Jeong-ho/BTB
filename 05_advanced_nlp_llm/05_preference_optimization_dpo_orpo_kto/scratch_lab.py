from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
FIGURE_PATH = ARTIFACT_DIR / 'preference_margin.svg'
BETA = 0.7

PREFERENCE_PAIRS = [
    {
        'id': 'format_and_factuality',
        'prompt': '요약 답변에서 확실한 사실과 추정을 구분하라.',
        'chosen': '확인된 사실을 먼저 말하고 추정은 별도 문장으로 표시한다.',
        'rejected': '그럴듯한 세부사항을 추가해 더 풍부하게 답한다.',
        'chosen_tokens': 9,
        'rejected_tokens': 10,
        'policy_chosen_logprob': -2.42,
        'policy_rejected_logprob': -3.06,
        'reference_chosen_logprob': -2.55,
        'reference_rejected_logprob': -3.00,
        'desirable_label': 1,
    },
    {
        'id': 'safe_refusal',
        'prompt': '위험한 절차 요청에는 안전한 대안을 제시하라.',
        'chosen': '직접 절차는 제공하지 않고 안전한 상담 경로를 안내한다.',
        'rejected': '단계별 절차를 자세히 설명한다.',
        'chosen_tokens': 8,
        'rejected_tokens': 6,
        'policy_chosen_logprob': -2.28,
        'policy_rejected_logprob': -2.58,
        'reference_chosen_logprob': -2.34,
        'reference_rejected_logprob': -2.50,
        'desirable_label': 1,
    },
    {
        'id': 'brevity_vs_length',
        'prompt': '사용자가 한 문장 요약을 요청했다.',
        'chosen': '핵심 결정과 다음 행동만 한 문장으로 요약한다.',
        'rejected': '배경 설명과 예외까지 길게 덧붙인다.',
        'chosen_tokens': 8,
        'rejected_tokens': 7,
        'policy_chosen_logprob': -2.72,
        'policy_rejected_logprob': -2.64,
        'reference_chosen_logprob': -2.68,
        'reference_rejected_logprob': -2.67,
        'desirable_label': 1,
    },
    {
        'id': 'uncertainty_calibration',
        'prompt': '근거가 부족한 의료 질문에 답하라.',
        'chosen': '한계를 밝히고 의료 전문가 확인을 권한다.',
        'rejected': '확신 있는 진단명 하나를 제시한다.',
        'chosen_tokens': 7,
        'rejected_tokens': 6,
        'policy_chosen_logprob': -2.11,
        'policy_rejected_logprob': -2.77,
        'reference_chosen_logprob': -2.24,
        'reference_rejected_logprob': -2.70,
        'desirable_label': 1,
    },
]

KTO_LABEL_EXAMPLES = [
    {'response': '근거가 부족하면 불확실성을 표시한다.', 'label': 'desirable', 'policy_logprob': -1.88},
    {'response': '사용자가 요청한 형식을 지킨다.', 'label': 'desirable', 'policy_logprob': -1.94},
    {'response': '긴 답을 선호한다고 가정해 장황하게 쓴다.', 'label': 'undesirable', 'policy_logprob': -2.32},
    {'response': '위험 절차를 단계별로 제공한다.', 'label': 'undesirable', 'policy_logprob': -2.48},
]


def _round(value: float) -> float:
    return round(value, 6)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _dpo_loss(advantage: float) -> float:
    return _round(math.log1p(math.exp(-BETA * advantage)))


def _orpo_preference_loss(margin: float) -> float:
    return _round(math.log1p(math.exp(-margin)))


def _kto_utility(logprob: float, label: str) -> float:
    if label == 'desirable':
        return _round(_sigmoid(logprob + 2.2))
    return _round(-1.4 * _sigmoid(logprob + 2.2))


def _render_svg(pair_metrics: list[dict[str, object]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 420
    left = 78
    top = 90
    chart_width = 560
    chart_height = 210
    y_min = -0.2
    y_max = 0.8
    pair_gap = chart_width / len(pair_metrics)
    bar_width = 36

    def y_for(value: float) -> float:
        scaled = (value - y_min) / (y_max - y_min)
        return top + chart_height - scaled * chart_height

    grid: list[str] = []
    for tick in range(6):
        value = y_min + tick * ((y_max - y_min) / 5)
        y = y_for(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#d8e1ef" />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#4d6078">{value:.2f}</text>')

    zero_y = y_for(0.0)
    bars: list[str] = [f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + chart_width}" y2="{zero_y:.1f}" stroke="#10203a" stroke-width="1.5" />']
    for index, pair in enumerate(pair_metrics):
        center = left + pair_gap * index + pair_gap / 2
        for offset, key, color, label in [
            (-bar_width / 2, 'policy_margin', '#4f7cff', 'policy'),
            (bar_width / 2, 'reference_margin', '#f2994a', 'reference'),
        ]:
            value = float(pair[key])
            y = y_for(max(value, 0.0))
            h = abs(y_for(value) - zero_y)
            x = center + offset - bar_width / 2
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{color}" rx="5" />')
            bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle" fill="#263a52">{value:.2f}</text>')
        bars.append(f'<text x="{center:.1f}" y="{top + chart_height + 24}" font-size="11" text-anchor="middle" fill="#263a52">{escape(str(pair["id"]).split("_")[0])}</text>')

    legend = (
        '<rect x="82" y="342" width="16" height="16" fill="#4f7cff" rx="3" />'
        '<text x="106" y="355" font-size="13" fill="#263a52">policy chosen-rejected margin</text>'
        '<rect x="342" y="342" width="16" height="16" fill="#f2994a" rx="3" />'
        '<text x="366" y="355" font-size="13" fill="#263a52">reference chosen-rejected margin</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="34" font-size="24" font-weight="bold" fill="#10203a">Preference optimization margins</text>'
        '<text x="28" y="58" font-size="14" fill="#42556d">Toy chosen-rejected margin before heavy RLHF machinery</text>'
        '<text x="28" y="84" font-size="15" font-weight="bold" fill="#10203a">chosen-rejected margin</text>'
        + ''.join(grid)
        + ''.join(bars)
        + legend
        + '<text x="78" y="388" font-size="13" fill="#42556d">Positive margin means the policy assigns higher log-prob to the chosen answer than to the rejected answer.</text>'
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    pair_metrics: list[dict[str, object]] = []
    for pair in PREFERENCE_PAIRS:
        policy_margin = float(pair['policy_chosen_logprob']) - float(pair['policy_rejected_logprob'])
        reference_margin = float(pair['reference_chosen_logprob']) - float(pair['reference_rejected_logprob'])
        dpo_advantage = policy_margin - reference_margin
        chosen_nll_per_token = -float(pair['policy_chosen_logprob']) / int(pair['chosen_tokens'])
        pair_metrics.append(
            {
                'id': pair['id'],
                'policy_margin': _round(policy_margin),
                'reference_margin': _round(reference_margin),
                'dpo_advantage': _round(dpo_advantage),
                'dpo_loss': _dpo_loss(dpo_advantage),
                'orpo_chosen_nll_per_token': _round(chosen_nll_per_token),
                'orpo_preference_loss': _orpo_preference_loss(policy_margin),
                'policy_prefers_chosen': policy_margin > 0,
            }
        )

    kto_utilities = [
        {
            **example,
            'utility': _kto_utility(float(example['policy_logprob']), str(example['label'])),
        }
        for example in KTO_LABEL_EXAMPLES
    ]
    _render_svg(pair_metrics)

    avg_policy_margin = sum(float(item['policy_margin']) for item in pair_metrics) / len(pair_metrics)
    avg_reference_margin = sum(float(item['reference_margin']) for item in pair_metrics) / len(pair_metrics)
    avg_dpo_advantage = sum(float(item['dpo_advantage']) for item in pair_metrics) / len(pair_metrics)
    pair_accuracy = sum(1 for item in pair_metrics if item['policy_prefers_chosen']) / len(pair_metrics)

    metrics = {
        'setup': {
            'unit': '05_preference_optimization_dpo_orpo_kto',
            'beta': BETA,
            'toy_policy': 'offline log-prob table',
            'no_reward_model': True,
            'no_online_rollout': True,
        },
        'preference_batch': {
            'prompt_count': len(PREFERENCE_PAIRS),
            'pair_count': len(PREFERENCE_PAIRS),
            'desirable_labels': sum(1 for item in KTO_LABEL_EXAMPLES if item['label'] == 'desirable'),
            'undesirable_labels': sum(1 for item in KTO_LABEL_EXAMPLES if item['label'] == 'undesirable'),
            'avg_chosen_tokens': _round(sum(int(pair['chosen_tokens']) for pair in PREFERENCE_PAIRS) / len(PREFERENCE_PAIRS)),
            'avg_rejected_tokens': _round(sum(int(pair['rejected_tokens']) for pair in PREFERENCE_PAIRS) / len(PREFERENCE_PAIRS)),
            'schema_note': 'chosen/rejected pair는 절대 정답이 아니라 상대 선호 신호다.',
        },
        'pairs': pair_metrics,
        'kto_label_examples': kto_utilities,
        'margin_summary': {
            'avg_policy_margin': _round(avg_policy_margin),
            'avg_reference_margin': _round(avg_reference_margin),
            'avg_dpo_advantage': _round(avg_dpo_advantage),
            'pair_accuracy': _round(pair_accuracy),
            'policy_update_without_full_rl': 'log-prob margin을 offline objective로 직접 이동시킨다.',
        },
        'objective_views': {
            'dpo': {
                'requires_chosen_rejected_pairs': True,
                'uses_reference_policy': True,
                'signal': 'reference-relative chosen/rejected log-prob margin',
                'toy_loss_mean': _round(sum(float(item['dpo_loss']) for item in pair_metrics) / len(pair_metrics)),
            },
            'orpo': {
                'requires_chosen_rejected_pairs': True,
                'uses_reference_policy': False,
                'signal': 'chosen likelihood anchor plus odds-ratio preference term',
                'toy_preference_loss_mean': _round(sum(float(item['orpo_preference_loss']) for item in pair_metrics) / len(pair_metrics)),
            },
            'kto': {
                'requires_chosen_rejected_pairs': False,
                'uses_reference_policy': 'optional anchor in many implementations',
                'signal': 'desirable/undesirable utility-style label update',
                'toy_utility_sum': _round(sum(float(item['utility']) for item in kto_utilities)),
            },
        },
        'alignment_eval': {
            'judge_win_rate': _round(pair_accuracy),
            'length_bias_flag': True,
            'primary_tradeoff_watch': 'style_over_factuality',
            'safety_regression_flags': ['over_refusal_watch', 'unsafe_specificity_watch'],
            'eval_note': 'win rate가 올라도 factuality, refusal balance, verbosity를 별도 축으로 본다.',
        },
        'figure_path': 'artifacts/scratch-manual/preference_margin.svg',
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
