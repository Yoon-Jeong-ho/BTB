from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'vqa_answer_type_accuracy.svg'
FEATURE_NAMES = ['red', 'blue', 'ball', 'cube', 'count_one', 'count_two']
ANSWER_TYPES = ('yes/no', 'color', 'count')


def build_toy_dataset() -> tuple[np.ndarray, list[dict[str, str]], list[str], list[str]]:
    image_features = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    questions = [
        {'question': '공은 빨간색인가?', 'answer_type': 'yes/no', 'target': 'ball', 'query_color': 'red'},
        {'question': '공은 빨간색인가?', 'answer_type': 'yes/no', 'target': 'ball', 'query_color': 'red'},
        {'question': '큐브 색은 무엇인가?', 'answer_type': 'color', 'target': 'cube', 'query_color': ''},
        {'question': '큐브는 몇 개인가?', 'answer_type': 'count', 'target': 'cube', 'query_color': ''},
        {'question': '큐브는 몇 개인가?', 'answer_type': 'count', 'target': 'cube', 'query_color': ''},
        {'question': '공 색은 무엇인가?', 'answer_type': 'color', 'target': 'ball', 'query_color': ''},
    ]
    gold_answers = ['yes', 'no', 'blue', '2', '1', 'blue']
    image_labels = [
        '빨간 공 한 개',
        '파란 공 한 개',
        '파란 큐브 한 개',
        '빨간 큐브 두 개',
        '빨간 큐브 한 개',
        '파란 공 두 개',
    ]
    return image_features, questions, gold_answers, image_labels


def _validate_inputs(
    image_features: np.ndarray,
    questions: list[dict[str, str]],
    gold_answers: list[str],
    image_labels: list[str],
) -> None:
    if image_features.ndim != 2:
        raise ValueError(
            'scratch visual question answering example expects a 2D image feature matrix shaped like '
            '(batch, feature_dim).'
        )
    batch_size = image_features.shape[0]
    if batch_size != len(questions) or batch_size != len(gold_answers) or batch_size != len(image_labels):
        raise ValueError(
            'image/question batch size must match for this visual question answering toy setup: '
            f'got images {batch_size}, questions {len(questions)}, answers {len(gold_answers)}, '
            f'labels {len(image_labels)}.'
        )


def _feature_value(feature_row: np.ndarray, name: str) -> float:
    return float(feature_row[FEATURE_NAMES.index(name)])


def _predict_yes_no(feature_row: np.ndarray, question: dict[str, str]) -> tuple[str, float, str]:
    target_color = question['query_color']
    positive_score = _feature_value(feature_row, target_color) + 0.35 * _feature_value(feature_row, question['target'])
    negative_score = 1.0 - _feature_value(feature_row, target_color)
    if positive_score >= negative_score:
        answer = 'yes'
        confidence = positive_score / max(positive_score + negative_score, 1e-8)
    else:
        answer = 'no'
        confidence = negative_score / max(positive_score + negative_score, 1e-8)
    return answer, round(float(confidence), 6), ''


def _predict_color(feature_row: np.ndarray) -> tuple[str, float, str]:
    red_score = 1.2 * _feature_value(feature_row, 'red')
    blue_score = 1.2 * _feature_value(feature_row, 'blue')
    if red_score >= blue_score:
        confidence = red_score / max(red_score + blue_score, 1e-8)
        return 'red', round(float(confidence), 6), ''
    confidence = blue_score / max(red_score + blue_score, 1e-8)
    return 'blue', round(float(confidence), 6), ''


def _predict_count(feature_row: np.ndarray) -> tuple[str, float, str]:
    one_score = 1.6 * _feature_value(feature_row, 'count_one') + 0.95 * _feature_value(feature_row, 'red') + 0.05
    two_score = 0.6 * _feature_value(feature_row, 'count_two') + 0.1 * _feature_value(feature_row, 'blue')
    if one_score >= two_score:
        confidence = one_score / max(one_score + two_score, 1e-8)
        return '1', round(float(confidence), 6), 'count_shortcut_prior'
    confidence = two_score / max(one_score + two_score, 1e-8)
    return '2', round(float(confidence), 6), ''


def predict_answer(feature_row: np.ndarray, question: dict[str, str]) -> tuple[str, float, str]:
    answer_type = question['answer_type']
    if answer_type == 'yes/no':
        return _predict_yes_no(feature_row, question)
    if answer_type == 'color':
        return _predict_color(feature_row)
    if answer_type == 'count':
        return _predict_count(feature_row)
    raise ValueError(f'Unsupported answer_type: {answer_type}')


def _compute_answer_type_accuracy(rows: list[dict[str, object]]) -> dict[str, float]:
    accuracy: dict[str, float] = {}
    for answer_type in ANSWER_TYPES:
        subset = [row for row in rows if row['answer_type'] == answer_type]
        if not subset:
            raise ValueError(f'Missing answer_type bucket for VQA accuracy: {answer_type}')
        accuracy[answer_type] = round(float(np.mean([row['is_correct'] for row in subset])), 6)
    return accuracy


def save_svg(answer_type_accuracy: dict[str, float], rows: list[dict[str, object]]) -> None:
    width, height = 840, 420
    left, top, bottom = 100, 80, 320
    bar_width = 100
    gap = 120
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        '  <text x="32" y="34" font-size="22" font-family="Arial, sans-serif">VQA answer-type accuracy (scratch)</text>',
        '  <text x="32" y="58" font-size="13" font-family="Arial, sans-serif" fill="#374151">yes/no, color, count별 정확도를 비교해 shortcut bias가 어디에 남는지 본다.</text>',
        f'  <line x1="{left}" y1="{bottom}" x2="740" y2="{bottom}" stroke="#111827" stroke-width="2" />',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" stroke-width="2" />',
    ]
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        svg_lines.append(f'  <line x1="{left - 6}" y1="{y:.1f}" x2="740" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1" />')
        svg_lines.append(
            f'  <text x="{left - 12}" y="{y + 4:.1f}" font-size="12" text-anchor="end" font-family="Arial, sans-serif">{value:.1f}</text>'
        )

    bar_colors = {'yes/no': '#2563eb', 'color': '#16a34a', 'count': '#dc2626'}
    for index, answer_type in enumerate(ANSWER_TYPES):
        accuracy = answer_type_accuracy[answer_type]
        bar_height = accuracy * (bottom - top)
        x = left + 50 + index * gap
        svg_lines.extend(
            [
                f'  <rect x="{x}" y="{bottom - bar_height:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{bar_colors[answer_type]}" rx="8" />',
                f'  <text x="{x + bar_width / 2:.1f}" y="{bottom - bar_height - 10:.1f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{accuracy:.2f}</text>',
                f'  <text x="{x + bar_width / 2:.1f}" y="{bottom + 24}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{escape(answer_type, quote=False)}</text>',
            ]
        )

    failure_examples = [row for row in rows if not row['is_correct']]
    failure_text = ' / '.join(
        f"{row['image_label']} → {row['predicted_answer']} (gold {row['gold_answer']})" for row in failure_examples
    ) or '오답 없음'
    svg_lines.extend(
        [
            f'  <text x="490" y="130" font-size="13" font-family="Arial, sans-serif" fill="#111827">오답 메모</text>',
            f'  <text x="490" y="156" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">{escape(failure_text, quote=False)}</text>',
            '</svg>',
        ]
    )
    FIGURE_PATH.write_text('\n'.join(svg_lines), encoding='utf-8')


def generate_vqa_metrics(
    image_features: np.ndarray,
    questions: list[dict[str, str]],
    gold_answers: list[str],
    image_labels: list[str],
) -> dict[str, object]:
    _validate_inputs(image_features, questions, gold_answers, image_labels)

    rows: list[dict[str, object]] = []
    for feature_row, question, gold_answer, image_label in zip(image_features, questions, gold_answers, image_labels):
        predicted_answer, confidence, error_reason = predict_answer(feature_row, question)
        is_correct = predicted_answer == gold_answer
        rows.append(
            {
                'image_label': image_label,
                'question': question['question'],
                'answer_type': question['answer_type'],
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'confidence': confidence,
                'error_reason': '' if is_correct else error_reason,
            }
        )

    answer_type_accuracy = _compute_answer_type_accuracy(rows)
    overall_accuracy = round(float(np.mean([row['is_correct'] for row in rows])), 6)
    shortcut_error_count = int(sum(row['error_reason'] == 'count_shortcut_prior' for row in rows))
    return {
        'sample_count': int(image_features.shape[0]),
        'feature_names': FEATURE_NAMES,
        'image_feature_shape': list(image_features.shape),
        'overall_accuracy': overall_accuracy,
        'answer_type_accuracy': answer_type_accuracy,
        'shortcut_error_count': shortcut_error_count,
        'rows': rows,
    }


def run() -> None:
    image_features, questions, gold_answers, image_labels = build_toy_dataset()
    metrics = generate_vqa_metrics(image_features, questions, gold_answers, image_labels)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(metrics['answer_type_accuracy'], metrics['rows'])
    metrics['figure_path'] = str(FIGURE_PATH.relative_to(UNIT_ROOT))

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
