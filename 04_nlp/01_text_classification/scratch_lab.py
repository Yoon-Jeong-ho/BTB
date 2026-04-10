from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'token_signal.svg'
LABELS = ['negative', 'positive']
TRAIN_ROWS = [
    ('회의 자료가 명확해서 이해가 빨랐다', 'positive'),
    ('배송이 늦고 포장이 찢어져서 실망했다', 'negative'),
    ('설명이 친절하고 예제가 좋아서 추천한다', 'positive'),
    ('앱이 자꾸 멈추고 광고가 너무 많다', 'negative'),
    ('수업 속도가 안정적이라 복습하기 좋았다', 'positive'),
    ('문장이 어색하고 번역 품질이 낮다', 'negative'),
    ('업데이트 후 검색이 빨라져서 만족한다', 'positive'),
    ('버튼이 안 눌리고 오류 메시지도 없었다', 'negative'),
]
EVAL_ROWS = [
    ('업데이트가 안정적이고 사용이 편하다', 'positive'),
    ('광고가 많고 실행이 느려 불편하다', 'negative'),
    ('예제가 친절해서 다시 보고 싶다', 'positive'),
    ('포장이 찢어지고 배송이 늦었다', 'negative'),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r'[가-힣A-Za-z0-9]+', text.lower())


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy_score(gold: list[str], pred: list[str]) -> float:
    matches = sum(1 for expected, actual in zip(gold, pred) if expected == actual)
    return _safe_div(matches, len(gold))


def macro_f1_score(gold: list[str], pred: list[str]) -> float:
    f1_values: list[float] = []
    for label in LABELS:
        tp = sum(1 for expected, actual in zip(gold, pred) if expected == label and actual == label)
        fp = sum(1 for expected, actual in zip(gold, pred) if expected != label and actual == label)
        fn = sum(1 for expected, actual in zip(gold, pred) if expected == label and actual != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
    return _safe_div(sum(f1_values), len(f1_values))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _save_svg(token_scores: dict[str, float]) -> None:
    width, height = 760, 420
    left, top = 160, 50
    bar_height = 28
    gap = 16
    mid_x = 380
    right_limit = 700
    max_abs = max((abs(score) for score in token_scores.values()), default=1.0)

    def scaled_width(score: float) -> float:
        return (abs(score) / max_abs) * (right_limit - mid_x - 30)

    rows = []
    sorted_items = sorted(token_scores.items(), key=lambda item: item[1])
    for index, (token, score) in enumerate(sorted_items):
        y = top + index * (bar_height + gap)
        width_value = scaled_width(score)
        if score >= 0:
            x = mid_x
            color = '#2b8a3e'
            anchor = 'start'
            text_x = 20
        else:
            x = mid_x - width_value
            color = '#d94841'
            anchor = 'end'
            text_x = mid_x - 20
        rows.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width_value:.2f}" height="{bar_height}" fill="{color}" opacity="0.85" />'
        )
        rows.append(
            f'<text x="{text_x}" y="{y + 19:.2f}" text-anchor="{anchor}" font-size="14" font-family="Arial, sans-serif">{token}</text>'
        )
        score_anchor = 'start' if score >= 0 else 'end'
        score_x = mid_x + 12 if score >= 0 else mid_x - 12
        rows.append(
            f'<text x="{score_x}" y="{y + 19:.2f}" text-anchor="{score_anchor}" font-size="12" font-family="Arial, sans-serif">{score:.3f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="32" y="30" font-size="22" font-family="Arial, sans-serif">Token signal by class (scratch)</text>
  <line x1="{mid_x}" y1="44" x2="{mid_x}" y2="{height - 30}" stroke="#555" stroke-width="2" />
  <text x="{mid_x - 18}" y="44" text-anchor="end" font-size="13" font-family="Arial, sans-serif" fill="#d94841">negative</text>
  <text x="{mid_x + 18}" y="44" text-anchor="start" font-size="13" font-family="Arial, sans-serif" fill="#2b8a3e">positive</text>
  {''.join(rows)}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    train_examples = [(tokenize(text), label) for text, label in TRAIN_ROWS]
    eval_examples = [(tokenize(text), label, text) for text, label in EVAL_ROWS]
    vocab = sorted({token for tokens, _ in train_examples for token in tokens})

    class_doc_counts = Counter(label for _, label in train_examples)
    class_token_counts = {label: Counter() for label in LABELS}
    total_tokens_by_class = Counter()
    for tokens, label in train_examples:
        class_token_counts[label].update(tokens)
        total_tokens_by_class[label] += len(tokens)

    vocab_size = len(vocab)
    class_log_prior = {
        label: math.log(class_doc_counts[label] / len(train_examples))
        for label in LABELS
    }

    predictions: list[str] = []
    prediction_rows: list[dict[str, object]] = []
    gold = [label for _, label, _ in eval_examples]
    for tokens, gold_label, text in eval_examples:
        class_scores: dict[str, float] = {}
        token_contributions: dict[str, dict[str, float]] = {label: {} for label in LABELS}
        for label in LABELS:
            score = class_log_prior[label]
            for token in tokens:
                token_log_prob = math.log(
                    (class_token_counts[label][token] + 1) / (total_tokens_by_class[label] + vocab_size)
                )
                score += token_log_prob
                token_contributions[label][token] = _rounded(token_log_prob)
            class_scores[label] = score

        predicted = max(class_scores, key=class_scores.get)
        predictions.append(predicted)
        prediction_rows.append(
            {
                'text': text,
                'gold': gold_label,
                'predicted': predicted,
                'tokenized': tokens,
                'score_margin': _rounded(class_scores['positive'] - class_scores['negative']),
                'positive_token_log_probs': token_contributions['positive'],
                'negative_token_log_probs': token_contributions['negative'],
            }
        )

    signal_scores = {}
    for token in vocab:
        positive_log_prob = math.log((class_token_counts['positive'][token] + 1) / (total_tokens_by_class['positive'] + vocab_size))
        negative_log_prob = math.log((class_token_counts['negative'][token] + 1) / (total_tokens_by_class['negative'] + vocab_size))
        signal_scores[token] = positive_log_prob - negative_log_prob

    top_positive = [token for token, _ in sorted(signal_scores.items(), key=lambda item: item[1], reverse=True)[:5]]
    top_negative = [token for token, _ in sorted(signal_scores.items(), key=lambda item: item[1])[:5]]
    plotted_scores = {token: _rounded(signal_scores[token]) for token in (top_negative[::-1] + top_positive)}

    confusion_matrix = {
        label: {predicted_label: 0 for predicted_label in LABELS}
        for label in LABELS
    }
    for expected, actual in zip(gold, predictions):
        confusion_matrix[expected][actual] += 1

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _save_svg(plotted_scores)

    metrics = {
        'train_size': len(train_examples),
        'eval_size': len(eval_examples),
        'vocab_size': vocab_size,
        'labels': LABELS,
        'class_priors': {label: _rounded(math.exp(class_log_prior[label])) for label in LABELS},
        'eval_accuracy': _rounded(accuracy_score(gold, predictions)),
        'eval_macro_f1': _rounded(macro_f1_score(gold, predictions)),
        'top_positive_tokens': top_positive,
        'top_negative_tokens': top_negative,
        'token_signal_scores': {token: _rounded(score) for token, score in signal_scores.items()},
        'confusion_matrix': confusion_matrix,
        'prediction_rows': prediction_rows,
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
