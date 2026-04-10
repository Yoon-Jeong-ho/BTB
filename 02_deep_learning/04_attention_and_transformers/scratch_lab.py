from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
FIGURE_PATH = ARTIFACT_DIR / 'attention_patterns.svg'
TOKENS = ['학생은', '어제', '도서관에서', '책을', '읽었다']
HEAD_SPECS = [
    {
        'name': 'local-syntax-head',
        'query': [
            [1.2, 0.2],
            [0.8, 0.6],
            [0.4, 1.0],
            [0.3, 1.1],
            [0.9, 0.7],
        ],
        'key': [
            [1.1, 0.1],
            [0.7, 0.5],
            [0.5, 0.9],
            [0.2, 1.2],
            [0.8, 0.8],
        ],
        'value': [
            [1.0, 0.1],
            [0.8, 0.3],
            [0.4, 0.9],
            [0.2, 1.0],
            [0.7, 0.8],
        ],
    },
    {
        'name': 'long-context-head',
        'query': [
            [0.3, 1.0],
            [1.0, 0.3],
            [1.1, 0.1],
            [0.2, 1.2],
            [1.2, 0.2],
        ],
        'key': [
            [0.1, 1.1],
            [1.1, 0.2],
            [1.0, 0.4],
            [0.3, 1.0],
            [1.2, 0.1],
        ],
        'value': [
            [0.2, 1.0],
            [1.0, 0.3],
            [0.9, 0.4],
            [0.3, 0.9],
            [1.1, 0.2],
        ],
    },
]
FOCUS_QUERY_INDEX = 2


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax(row: list[float]) -> list[float]:
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def masked_softmax(row: list[float], *, query_index: int, causal: bool) -> list[float]:
    if not causal:
        return softmax(row)

    masked = []
    for key_index, value in enumerate(row):
        masked.append(value if key_index <= query_index else float('-inf'))
    finite_values = [value for value in masked if value != float('-inf')]
    max_value = max(finite_values)
    exps = [math.exp(value - max_value) if value != float('-inf') else 0.0 for value in masked]
    total = sum(exps)
    return [value / total for value in exps]


def weighted_sum(weights: list[float], values: list[list[float]]) -> list[float]:
    dims = len(values[0]) if values else 0
    mixed = []
    for dim_index in range(dims):
        mixed.append(sum(weight * value[dim_index] for weight, value in zip(weights, values)))
    return mixed


def round_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(value, 6) for value in row] for row in matrix]


def build_head_metrics(spec: dict[str, object]) -> dict[str, object]:
    query = spec['query']
    key = spec['key']
    value = spec['value']
    assert isinstance(query, list)
    assert isinstance(key, list)
    assert isinstance(value, list)
    scale = math.sqrt(len(key[0]))
    scores = [[dot(q, k) / scale for k in key] for q in query]
    weights = [softmax(row) for row in scores]
    mixed_outputs = [weighted_sum(row, value) for row in weights]
    top_links = []
    for query_token, row in zip(TOKENS, weights):
        top_index = max(range(len(row)), key=row.__getitem__)
        top_links.append(
            {
                'query_token': query_token,
                'top_key_token': TOKENS[top_index],
                'top_weight': round(row[top_index], 6),
            }
        )
    return {
        'name': spec['name'],
        'row_sums': [round(sum(row), 6) for row in weights],
        'raw_scores': round_matrix(scores),
        'attention_weights': round_matrix(weights),
        'mixed_outputs': round_matrix(mixed_outputs),
        'top_links': top_links,
        'focus_query_weights': {
            token: round(weight, 6)
            for token, weight in zip(TOKENS, weights[FOCUS_QUERY_INDEX])
        },
    }


def render_heatmap_svg(matrix: list[list[float]], labels: list[str]) -> None:
    cell = 56
    margin = 92
    width = margin + len(labels) * cell + 24
    height = margin + len(labels) * cell + 48
    rows = []
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            shade = 255 - int(round(value * 180))
            color = f'rgb({shade}, {shade}, 255)'
            x = margin + col_index * cell
            y = margin + row_index * cell
            rows.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#5b6b8a" />'
            )
            rows.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" '
                'font-size="12" text-anchor="middle" fill="#10203a">'
                f'{value:.2f}</text>'
            )

    label_nodes = []
    for index, label in enumerate(labels):
        safe = escape(label)
        label_nodes.append(
            f'<text x="{margin + index * cell + cell / 2}" y="68" font-size="12" '
            f'text-anchor="middle" fill="#10203a">{safe}</text>'
        )
        label_nodes.append(
            f'<text x="62" y="{margin + index * cell + cell / 2 + 5}" font-size="12" '
            f'text-anchor="middle" fill="#10203a">{safe}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="24" y="28" font-size="20" font-weight="bold" fill="#10203a">'
        'Attention pattern heatmap</text>'
        '<text x="24" y="48" font-size="12" fill="#36445c">'
        'head 0 attention weights (query rows × key columns)</text>'
        + ''.join(label_nodes)
        + ''.join(rows)
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    heads = [build_head_metrics(spec) for spec in HEAD_SPECS]
    row_sum_errors = [abs(row_sum - 1.0) for head in heads for row_sum in head['row_sums']]

    encoder_scores = heads[0]['raw_scores']
    encoder_weights = [softmax(row) for row in encoder_scores]
    decoder_weights = [
        masked_softmax(row, query_index=query_index, causal=True)
        for query_index, row in enumerate(encoder_scores)
    ]
    future_mass_encoder = sum(
        encoder_weights[FOCUS_QUERY_INDEX][key_index]
        for key_index in range(FOCUS_QUERY_INDEX + 1, len(TOKENS))
    )
    future_mass_decoder = sum(
        decoder_weights[FOCUS_QUERY_INDEX][key_index]
        for key_index in range(FOCUS_QUERY_INDEX + 1, len(TOKENS))
    )

    distinct_top_key_counts = []
    for query_index, token in enumerate(TOKENS):
        top_keys = [head['top_links'][query_index]['top_key_token'] for head in heads]
        distinct_top_key_counts.append(
            {
                'query_token': token,
                'top_keys': top_keys,
                'distinct_top_key_count': len(set(top_keys)),
            }
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    render_heatmap_svg(encoder_weights, TOKENS)

    metrics = {
        'tokens': TOKENS,
        'sequence_length': len(TOKENS),
        'max_row_sum_error': round(max(row_sum_errors), 10),
        'single_head_row_sums': [round(sum(row), 6) for row in encoder_weights],
        'multi_head': {
            'head_count': len(heads),
            'heads': heads,
            'distinct_top_key_counts': [item['distinct_top_key_count'] for item in distinct_top_key_counts],
            'per_query_top_keys': distinct_top_key_counts,
        },
        'encoder_decoder': {
            'focus_query_token': TOKENS[FOCUS_QUERY_INDEX],
            'encoder_future_access_mass': round(future_mass_encoder, 6),
            'decoder_future_access_mass': round(future_mass_decoder, 6),
            'encoder_top_key': heads[0]['top_links'][FOCUS_QUERY_INDEX]['top_key_token'],
            'decoder_top_key': TOKENS[
                max(
                    range(len(decoder_weights[FOCUS_QUERY_INDEX])),
                    key=decoder_weights[FOCUS_QUERY_INDEX].__getitem__,
                )
            ],
            'causal_mask_future_blocked': future_mass_decoder < 1e-9,
        },
        'recurrent_relief': {
            'recurrent_steps': len(TOKENS),
            'attention_parallel_rounds': 1,
            'longest_dependency_path_rnn': len(TOKENS) - 1,
            'longest_dependency_path_attention': 1,
            'pairwise_score_count': len(TOKENS) ** 2,
        },
        'figure_path': 'artifacts/scratch-manual/attention_patterns.svg',
        'sequence_mixing_summary': (
            'attention output은 토큰 하나를 뽑는 것이 아니라 row weight로 value들을 섞은 결과이며, '
            'decoder에서는 causal mask가 미래 mixing을 막는다.'
        ),
    }

    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
