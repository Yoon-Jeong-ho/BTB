from __future__ import annotations

import json
import math
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
TOKENS = ['나는', '커피를', '정말', '좋아해요']
QUERY = [
    [1.2, 0.1],
    [0.2, 1.3],
    [0.6, 0.8],
    [1.0, 0.9],
]
KEY = [
    [1.1, 0.2],
    [0.1, 1.2],
    [0.5, 0.7],
    [1.0, 0.8],
]
VALUE = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.4, 0.6],
    [0.9, 0.8],
]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(row: list[float]) -> list[float]:
    max_value = max(row)
    exps = [math.exp(value - max_value) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def _weighted_sum(weights: list[float], values: list[list[float]]) -> list[float]:
    dims = len(values[0]) if values else 0
    mixed = []
    for dim_index in range(dims):
        mixed.append(sum(weight * value[dim_index] for weight, value in zip(weights, values)))
    return mixed


def _round_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(value, 6) for value in row] for row in matrix]


def run() -> None:
    scale = math.sqrt(len(KEY[0]))
    scores = [[_dot(query, key) / scale for key in KEY] for query in QUERY]
    attention_weights = [_softmax(row) for row in scores]
    mixed_outputs = [_weighted_sum(weights, VALUE) for weights in attention_weights]

    strongest_links = []
    for query_token, weights in zip(TOKENS, attention_weights):
        strongest_index = max(range(len(weights)), key=weights.__getitem__)
        strongest_links.append(
            {
                'query_token': query_token,
                'top_key_token': TOKENS[strongest_index],
                'top_weight': round(weights[strongest_index], 6),
            }
        )

    focus_index = TOKENS.index('좋아해요')
    metrics = {
        'tokens': TOKENS,
        'sequence_length': len(TOKENS),
        'hidden_dim': len(KEY[0]),
        'score_scale': round(scale, 6),
        'raw_scores': _round_matrix(scores),
        'attention_weights': _round_matrix(attention_weights),
        'row_sums': [round(sum(row), 6) for row in attention_weights],
        'mixed_outputs': _round_matrix(mixed_outputs),
        'strongest_links': strongest_links,
        'focus_query_token': TOKENS[focus_index],
        'focus_query_weights': {
            token: round(weight, 6)
            for token, weight in zip(TOKENS, attention_weights[focus_index])
        },
        'focus_query_mixed_output': [round(value, 6) for value in mixed_outputs[focus_index]],
        'sequence_mixing_explanation': (
            'attention output은 한 토큰을 그대로 복사한 값이 아니라, 각 key 위치 weight로 value를 가중합해 섞은 결과다.'
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
