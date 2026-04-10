from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
FIGURE_PATH = ARTIFACT_DIR / 'objective_comparison.svg'
BASE_SEQUENCE = ['<bos>', '연구자는', '긴', '문맥을', '천천히', '읽는다', '<eos>']
CONTEXT_WINDOW = 4
MASK_POSITIONS = [3, 4]
CORRUPTED_SPAN = ['문맥을', '천천히']


def _round(value: float) -> float:
    return round(value, 6)


def _loss_density(scored_tokens: int, total_slots: int) -> float:
    return _round(scored_tokens / total_slots)


def _render_svg(densities: dict[str, float]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    width = 640
    height = 360
    chart_top = 90
    chart_height = 180
    bar_width = 110
    gap = 70
    left = 70
    colors = {
        'causal_lm': '#4f7cff',
        'masked_lm': '#34a853',
        'span_corruption': '#f2994a',
    }
    labels = {
        'causal_lm': 'causal LM',
        'masked_lm': 'masked LM',
        'span_corruption': 'span corruption',
    }

    bars: list[str] = []
    for index, (name, density) in enumerate(densities.items()):
        x = left + index * (bar_width + gap)
        bar_height = density * chart_height
        y = chart_top + (chart_height - bar_height)
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
            f'fill="{colors[name]}" rx="10" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" font-size="14" text-anchor="middle" '
            f'fill="#10203a">{density:.3f}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 26}" font-size="14" '
            f'text-anchor="middle" fill="#10203a">{escape(labels[name])}</text>'
        )

    ticks: list[str] = []
    for tick in range(6):
        value = tick / 5
        y = chart_top + chart_height - value * chart_height
        ticks.append(
            f'<line x1="52" y1="{y:.1f}" x2="560" y2="{y:.1f}" stroke="#d7e0f0" />'
        )
        ticks.append(
            f'<text x="42" y="{y + 4:.1f}" font-size="12" text-anchor="end" fill="#4c6078">'
            f'{value:.1f}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="34" font-size="24" font-weight="bold" fill="#10203a">'
        'Pretraining objective comparison</text>'
        '<text x="28" y="58" font-size="14" fill="#42556d">'
        'Loss mask density with the same toy sentence and context window</text>'
        '<text x="28" y="84" font-size="15" font-weight="bold" fill="#10203a">'
        'Loss mask density</text>'
        + ''.join(ticks)
        + ''.join(bars)
        + '<text x="28" y="326" font-size="13" fill="#42556d">'
        'Same context window(4) does not mean same visible context: target framing changes what is seen and scored.</text>'
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    total_slots = len(BASE_SEQUENCE) - 1

    causal_input = BASE_SEQUENCE[:-1]
    causal_targets = BASE_SEQUENCE[1:]
    causal_focus_visible = causal_input[max(0, len(causal_input) - CONTEXT_WINDOW - 1) : -1]

    masked_input = BASE_SEQUENCE.copy()
    for position in MASK_POSITIONS:
        masked_input[position] = '[MASK]'
    masked_targets: list[str | None] = [None] * len(BASE_SEQUENCE)
    for position in MASK_POSITIONS:
        masked_targets[position] = BASE_SEQUENCE[position]
    masked_focus_visible = masked_input[1:6]

    span_encoder_input = ['연구자는', '긴', '<extra_id_0>', '읽는다', '<eos>']
    span_decoder_target = ['<extra_id_0>', *CORRUPTED_SPAN, '<extra_id_1>']

    objectives = {
        'causal_lm': {
            'input_tokens': causal_input,
            'target_tokens': causal_targets,
            'loss_mask': [1] * len(causal_targets),
            'scored_tokens': len(causal_targets),
            'loss_mask_density': _loss_density(len(causal_targets), total_slots),
            'target_framing': 'next-token prediction',
            'visible_context': 'left-only within the window',
            'focus_example': {
                'predict_token': '읽는다',
                'visible_tokens': causal_focus_visible,
                'future_visible': False,
            },
        },
        'masked_lm': {
            'input_tokens': masked_input,
            'target_tokens': masked_targets,
            'loss_mask': [1 if index in MASK_POSITIONS else 0 for index in range(len(BASE_SEQUENCE))],
            'scored_tokens': len(MASK_POSITIONS),
            'loss_mask_density': _loss_density(len(MASK_POSITIONS), total_slots),
            'target_framing': 'recover masked tokens only',
            'visible_context': 'bidirectional context around masked slots',
            'mask_positions': MASK_POSITIONS,
            'focus_example': {
                'predict_token': '문맥을',
                'visible_tokens': masked_focus_visible,
                'future_visible': True,
            },
        },
        'span_corruption': {
            'encoder_input_tokens': span_encoder_input,
            'decoder_target_tokens': span_decoder_target,
            'loss_mask': [1] * len(span_decoder_target),
            'scored_tokens': len(span_decoder_target),
            'loss_mask_density': _loss_density(len(span_decoder_target), total_slots),
            'target_framing': 'decoder reconstructs missing span',
            'visible_context': 'encoder reads corrupted document, decoder reads previous targets only',
            'corrupted_span': CORRUPTED_SPAN,
            'focus_example': {
                'predict_token': '문맥을',
                'encoder_visible_tokens': span_encoder_input,
                'decoder_prefix': ['<extra_id_0>'],
                'future_visible': False,
            },
        },
    }

    densities = {name: objective['loss_mask_density'] for name, objective in objectives.items()}
    density_ranking = [
        name for name, _ in sorted(densities.items(), key=lambda item: (-item[1], item[0]))
    ]
    _render_svg(densities)

    metrics = {
        'base_sequence': BASE_SEQUENCE,
        'context_window_tokens': CONTEXT_WINDOW,
        'objectives': objectives,
        'comparisons': {
            'density_ranking': density_ranking,
            'densest_supervision': density_ranking[0],
            'sparsest_supervision': density_ranking[-1],
            'causal_future_blocked': True,
            'masked_middle_token_sees_both_sides': True,
            'span_decoder_reads_previous_targets_only': True,
            'context_window_note': (
                '같은 context window=4라도 target framing이 바뀌면 visible context와 loss 위치가 달라진다.'
            ),
        },
        'figure_path': 'artifacts/scratch-manual/objective_comparison.svg',
    }

    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
