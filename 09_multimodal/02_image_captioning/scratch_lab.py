from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'caption_diagnostics.svg'
FEATURE_NAMES = ['cat', 'dog', 'kite', 'soup', 'mat', 'beach', 'bowl']
CONTENT_TOKENS = {'cat', 'dog', 'kite', 'bowl', 'mat', 'beach', 'soup'}


def build_toy_dataset() -> tuple[np.ndarray, list[list[str]], list[str]]:
    image_features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    references = [
        ['a', 'cat', 'on', 'mat'],
        ['a', 'kite', 'over', 'beach'],
        ['a', 'bowl', 'of', 'soup'],
        ['a', 'dog', 'on', 'beach'],
    ]
    image_labels = [
        '실내 고양이 매트',
        '해변 위 연',
        '수프가 담긴 그릇',
        '해변을 걷는 강아지',
    ]
    return image_features, references, image_labels


def _validate_caption_inputs(
    image_features: np.ndarray,
    references: list[list[str]],
    image_labels: list[str],
) -> None:
    if image_features.ndim != 2:
        raise ValueError(
            'scratch image captioning example expects a 2D image feature matrix shaped like '
            '(batch, feature_dim).'
        )
    batch_size = image_features.shape[0]
    if batch_size != len(references) or batch_size != len(image_labels):
        raise ValueError(
            'image/reference batch size must match for this image captioning toy setup: '
            f'got images {batch_size}, references {len(references)}, labels {len(image_labels)}.'
        )


def _feature_value(feature_row: np.ndarray, name: str) -> float:
    return float(feature_row[FEATURE_NAMES.index(name)])


def _select_subject(feature_row: np.ndarray) -> tuple[str, dict[str, float]]:
    scores = {
        'cat': 2.8 * _feature_value(feature_row, 'cat') + 1.1 * _feature_value(feature_row, 'mat'),
        'dog': 2.1 * _feature_value(feature_row, 'dog') + 1.4 * _feature_value(feature_row, 'beach') + 0.6,
        'kite': 1.5 * _feature_value(feature_row, 'kite') + 0.3 * _feature_value(feature_row, 'beach'),
        'bowl': 2.0 * _feature_value(feature_row, 'bowl') + 0.9 * _feature_value(feature_row, 'soup'),
    }
    subject = max(scores, key=scores.get)
    return subject, scores


def decode_caption(feature_row: np.ndarray) -> tuple[list[str], dict[str, float]]:
    subject, subject_scores = _select_subject(feature_row)
    if subject == 'bowl':
        relation = 'of'
        tail = 'soup'
    elif subject == 'kite':
        relation = 'over'
        tail = 'beach'
    else:
        relation = 'on'
        tail = 'beach' if _feature_value(feature_row, 'beach') >= _feature_value(feature_row, 'mat') else 'mat'

    caption_tokens = ['a', subject, relation, tail]
    diagnostics = {
        'subject_margin': round(
            float(sorted(subject_scores.values(), reverse=True)[0] - sorted(subject_scores.values(), reverse=True)[1]),
            6,
        ),
        'subject_scores': {key: round(float(value), 6) for key, value in subject_scores.items()},
    }
    return caption_tokens, diagnostics


def _content_overlap(reference: list[str], generated: list[str]) -> tuple[int, list[str], list[str]]:
    reference_content = [token for token in reference if token in CONTENT_TOKENS]
    generated_content = [token for token in generated if token in CONTENT_TOKENS]
    overlap = sum(token in reference_content for token in generated_content)
    missing = [token for token in reference_content if token not in generated_content]
    hallucinated = [token for token in generated_content if token not in reference_content]
    return overlap, missing, hallucinated


def corpus_unigram_precision(rows: list[dict[str, object]]) -> float:
    overlap = sum(int(row['content_overlap']) for row in rows)
    generated_total = sum(len(row['generated_content_tokens']) for row in rows)
    return round(float(overlap / generated_total), 6)


def mean_content_recall(rows: list[dict[str, object]]) -> float:
    recalls = []
    for row in rows:
        reference_total = len(row['reference_content_tokens'])
        if reference_total == 0:
            recalls.append(1.0)
        else:
            recalls.append(float(row['content_overlap']) / reference_total)
    return round(float(np.mean(recalls)), 6)


def save_svg(rows: list[dict[str, object]]) -> None:
    width, height = 860, 420
    left, right = 90, 760
    top, bottom = 80, 320
    bar_width = 48
    group_gap = 120
    max_length = max(int(row['generated_length']) for row in rows)
    max_hallucination = max(int(row['hallucinated_count']) for row in rows)
    max_value = max(max_length, max_hallucination + 1)

    def map_y(value: float) -> float:
        return bottom - (value / max_value) * (bottom - top)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        '  <text x="32" y="34" font-size="22" font-family="Arial, sans-serif">Caption diagnostics (scratch)</text>',
        '  <text x="32" y="58" font-size="13" font-family="Arial, sans-serif" fill="#374151">파란 막대는 생성 길이, 빨간 막대는 hallucination content token 수다.</text>',
        f'  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" stroke-width="2" />',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" stroke-width="2" />',
    ]

    for tick in range(max_value + 1):
        y = map_y(tick)
        svg_lines.append(
            f'  <line x1="{left - 6}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        svg_lines.append(
            f'  <text x="{left - 18}" y="{y + 4:.1f}" font-size="12" text-anchor="end" font-family="Arial, sans-serif">{tick}</text>'
        )

    for index, row in enumerate(rows):
        group_left = left + index * group_gap + 30
        length_height = bottom - map_y(float(row['generated_length']))
        hallucination_height = bottom - map_y(float(row['hallucinated_count']))
        label = escape(str(row['image_label']), quote=False)
        caption = escape(str(row['generated_caption']), quote=False)

        svg_lines.extend(
            [
                f'  <rect x="{group_left}" y="{bottom - length_height:.1f}" width="{bar_width}" height="{length_height:.1f}" fill="#2563eb" rx="6" />',
                f'  <rect x="{group_left + 58}" y="{bottom - hallucination_height:.1f}" width="{bar_width}" height="{hallucination_height:.1f}" fill="#dc2626" rx="6" />',
                f'  <text x="{group_left + bar_width / 2:.1f}" y="{bottom - length_height - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">{row["generated_length"]}</text>',
                f'  <text x="{group_left + 58 + bar_width / 2:.1f}" y="{bottom - hallucination_height - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">{row["hallucinated_count"]}</text>',
                f'  <text x="{group_left + 52:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">샘플 {index + 1}</text>',
                f'  <text x="{group_left + 52:.1f}" y="{bottom + 42}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#374151">{label}</text>',
                f'  <text x="{group_left + 52:.1f}" y="{bottom + 64}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#6b7280">{caption}</text>',
            ]
        )

    svg_lines.extend(
        [
            '  <rect x="620" y="92" width="12" height="12" fill="#2563eb" />',
            '  <text x="640" y="102" font-size="12" font-family="Arial, sans-serif">generated length</text>',
            '  <rect x="620" y="116" width="12" height="12" fill="#dc2626" />',
            '  <text x="640" y="126" font-size="12" font-family="Arial, sans-serif">hallucinated content tokens</text>',
            '</svg>',
        ]
    )
    FIGURE_PATH.write_text('\n'.join(svg_lines), encoding='utf-8')


def generate_caption_metrics(
    image_features: np.ndarray,
    references: list[list[str]],
    image_labels: list[str],
) -> dict[str, object]:
    _validate_caption_inputs(image_features, references, image_labels)

    rows: list[dict[str, object]] = []
    for feature_row, reference, image_label in zip(image_features, references, image_labels):
        generated_tokens, diagnostics = decode_caption(feature_row)
        overlap, missing, hallucinated = _content_overlap(reference, generated_tokens)
        reference_content_tokens = [token for token in reference if token in CONTENT_TOKENS]
        generated_content_tokens = [token for token in generated_tokens if token in CONTENT_TOKENS]
        rows.append(
            {
                'image_label': image_label,
                'reference_caption': ' '.join(reference),
                'generated_caption': ' '.join(generated_tokens),
                'reference_content_tokens': reference_content_tokens,
                'generated_content_tokens': generated_content_tokens,
                'content_overlap': overlap,
                'missing_content_tokens': missing,
                'hallucinated_content_tokens': hallucinated,
                'hallucinated_count': len(hallucinated),
                'generated_length': len(generated_tokens),
                'is_exact_match': generated_tokens == reference,
                'subject_margin': diagnostics['subject_margin'],
                'subject_scores': diagnostics['subject_scores'],
            }
        )

    exact_match_rate = round(float(np.mean([row['is_exact_match'] for row in rows])), 6)
    hallucinated_total = int(sum(int(row['hallucinated_count']) for row in rows))

    return {
        'sample_count': int(image_features.shape[0]),
        'feature_names': FEATURE_NAMES,
        'image_feature_shape': list(image_features.shape),
        'exact_match_rate': exact_match_rate,
        'corpus_unigram_precision': corpus_unigram_precision(rows),
        'mean_content_recall': mean_content_recall(rows),
        'mean_caption_length': round(float(np.mean([row['generated_length'] for row in rows])), 6),
        'hallucinated_content_tokens_total': hallucinated_total,
        'rows': rows,
    }


def run() -> None:
    image_features, references, image_labels = build_toy_dataset()
    metrics = generate_caption_metrics(image_features, references, image_labels)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(metrics['rows'])
    metrics['figure_path'] = str(FIGURE_PATH.relative_to(UNIT_ROOT))

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
