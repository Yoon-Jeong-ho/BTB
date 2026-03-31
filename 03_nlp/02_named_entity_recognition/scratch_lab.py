from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'label_distribution.svg'
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

TRAIN_ROWS = [
    {'words': ['김민수', '는', '서울', '시청', '에서', '일한다'], 'tags': ['B-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']},
    {'words': ['이서연', '은', '네이버', '클라우드', '팀에', '합류했다'], 'tags': ['B-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'O']},
    {'words': ['부산', '항만', '공사는', '오늘', '회의를', '열었다'], 'tags': ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O']},
    {'words': ['카카오', '직원들이', '판교', '오피스에서', '발표했다'], 'tags': ['B-ORG', 'O', 'B-LOC', 'O', 'O']},
    {'words': ['박지훈', '교수는', '연세대', '세미나에', '참석했다'], 'tags': ['B-PER', 'O', 'B-ORG', 'O', 'O']},
    {'words': ['LG', '에너지', '솔루션은', '서울', '본사를', '옮겼다'], 'tags': ['B-ORG', 'I-ORG', 'O', 'B-LOC', 'O', 'O']},
    {'words': ['정유진', '기자는', '제주', '공항을', '취재했다'], 'tags': ['B-PER', 'O', 'B-LOC', 'I-LOC', 'O']},
    {'words': ['한화', '시스템', '직원이', '부산', '항만을', '방문했다'], 'tags': ['B-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'O']},
]
EVAL_ROWS = [
    {'words': ['최민호', '는', '서울', '본사로', '출근했다'], 'tags': ['B-PER', 'O', 'B-LOC', 'O', 'O']},
    {'words': ['네이버', '클라우드', '직원이', '부산', '항만을', '방문했다'], 'tags': ['B-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'O']},
    {'words': ['카카오', '직원', '이서연이', '판교', '오피스에', '모였다'], 'tags': ['B-ORG', 'O', 'B-PER', 'B-LOC', 'O', 'O']},
    {'words': ['박지훈', '은', '제주', '공항', '행사에', '참석했다'], 'tags': ['B-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']},
]


def split_wordpieces(word: str) -> list[str]:
    if len(word) <= 2:
        return [word]
    return [word[:2], f'##{word[2:]}']


def align_word_tags(words: list[str], tags: list[str]) -> tuple[list[str], list[str], list[list[int]]]:
    pieces: list[str] = []
    aligned_tags: list[str] = []
    alignment: list[list[int]] = []
    for word_index, (word, tag) in enumerate(zip(words, tags)):
        word_pieces = split_wordpieces(word)
        alignment.append([])
        entity_type = tag.split('-', 1)[1] if '-' in tag else ''
        for piece_index, piece in enumerate(word_pieces):
            pieces.append(piece)
            alignment[-1].append(len(pieces) - 1)
            if tag == 'O':
                aligned_tags.append('O')
            elif piece_index == 0:
                aligned_tags.append(tag)
            else:
                aligned_tags.append(f'I-{entity_type}')
    return pieces, aligned_tags, alignment


def normalize_label(label: str) -> str:
    if label in LABELS:
        return label
    if label.startswith('I-'):
        converted = f'B-{label.split('-', 1)[1]}'
        if converted in LABELS:
            return converted
    return 'O'


def repair_bio(tags: list[str]) -> list[str]:
    repaired: list[str] = []
    previous_type: str | None = None
    previous_inside = False
    for tag in tags:
        normalized = normalize_label(tag)
        if normalized == 'O':
            repaired.append('O')
            previous_type = None
            previous_inside = False
            continue
        prefix, entity_type = normalized.split('-', 1)
        if prefix == 'I' and (not previous_inside or previous_type != entity_type):
            candidate = f'B-{entity_type}'
            normalized = candidate if candidate in LABELS else 'O'
            prefix = normalized.split('-', 1)[0] if normalized != 'O' else 'O'
        repaired.append(normalized)
        if normalized == 'O':
            previous_type = None
            previous_inside = False
        else:
            previous_type = entity_type
            previous_inside = prefix in {'B', 'I'}
    return repaired


def decode_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    entities: list[tuple[str, int, int]] = []
    start: int | None = None
    entity_type: str | None = None
    for index, tag in enumerate(repair_bio(tags) + ['O']):
        if tag == 'O':
            if start is not None and entity_type is not None:
                entities.append((entity_type, start, index))
                start = None
                entity_type = None
            continue
        prefix, current_type = tag.split('-', 1)
        if prefix == 'B' or entity_type != current_type:
            if start is not None and entity_type is not None:
                entities.append((entity_type, start, index))
            start = index
            entity_type = current_type
        elif start is None:
            start = index
            entity_type = current_type
    return entities


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float) -> float:
    return round(float(value), 6)


def token_accuracy(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> float:
    gold_flat = [tag for sequence in gold_sequences for tag in sequence]
    pred_flat = [tag for sequence in pred_sequences for tag in sequence]
    correct = sum(1 for gold, pred in zip(gold_flat, pred_flat) if gold == pred)
    return safe_div(correct, len(gold_flat))


def entity_metrics(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, float]:
    gold_entities = []
    pred_entities = []
    for index, (gold_tags, pred_tags) in enumerate(zip(gold_sequences, pred_sequences)):
        gold_entities.extend((index, *entity) for entity in decode_entities(gold_tags))
        pred_entities.extend((index, *entity) for entity in decode_entities(pred_tags))

    gold_set = set(gold_entities)
    pred_set = set(pred_entities)
    true_positive = len(gold_set & pred_set)
    precision = safe_div(true_positive, len(pred_set))
    recall = safe_div(true_positive, len(gold_set))
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        'entity_precision': rounded(precision),
        'entity_recall': rounded(recall),
        'entity_f1': rounded(f1),
    }


def build_piece_majority(rows: list[dict[str, list[str]]]) -> dict[str, str]:
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        pieces, aligned_tags, _ = align_word_tags(row['words'], row['tags'])
        for piece, tag in zip(pieces, aligned_tags):
            label_counts[piece][normalize_label(tag)] += 1
    return {piece: counts.most_common(1)[0][0] for piece, counts in label_counts.items()}


def label_counts(rows: list[dict[str, list[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        _, aligned_tags, _ = align_word_tags(row['words'], row['tags'])
        counts.update(normalize_label(tag) for tag in aligned_tags)
    return counts


def save_svg(counts: Counter[str]) -> None:
    width, height = 720, 360
    padding_left = 90
    padding_bottom = 60
    chart_height = 220
    chart_width = 600
    bar_width = 58
    gap = 12
    max_count = max(counts.values(), default=1)
    colors = {
        'O': '#868e96',
        'B-PER': '#1c7ed6',
        'I-PER': '#74c0fc',
        'B-ORG': '#5f3dc4',
        'I-ORG': '#9775fa',
        'B-LOC': '#2b8a3e',
        'I-LOC': '#74b816',
    }

    bars = []
    for index, label in enumerate(LABELS):
        x = padding_left + index * (bar_width + gap)
        bar_h = chart_height * (counts.get(label, 0) / max_count)
        y = 40 + chart_height - bar_h
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_h:.2f}" fill="{colors[label]}" opacity="0.88" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{40 + chart_height + 24}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{label}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{counts.get(label, 0)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="32" y="30" font-size="22" font-family="Arial, sans-serif">Aligned BIO label distribution (scratch)</text>
  <line x1="{padding_left - 20}" y1="{40 + chart_height}" x2="{padding_left + chart_width}" y2="{40 + chart_height}" stroke="#495057" stroke-width="2" />
  <line x1="{padding_left - 20}" y1="40" x2="{padding_left - 20}" y2="{40 + chart_height}" stroke="#495057" stroke-width="2" />
  {''.join(bars)}
  <text x="28" y="58" font-size="12" font-family="Arial, sans-serif" fill="#495057">count</text>
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    piece_label_map = build_piece_majority(TRAIN_ROWS)
    train_counts = label_counts(TRAIN_ROWS)

    gold_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    prediction_rows: list[dict[str, object]] = []

    for row in EVAL_ROWS:
        pieces, gold_tags, alignment = align_word_tags(row['words'], row['tags'])
        predicted = [piece_label_map.get(piece, 'O') for piece in pieces]
        repaired = repair_bio(predicted)
        gold_sequences.append([normalize_label(tag) for tag in gold_tags])
        pred_sequences.append(repaired)
        prediction_rows.append(
            {
                'words': row['words'],
                'word_tags': row['tags'],
                'pieces': pieces,
                'gold_piece_tags': gold_tags,
                'predicted_piece_tags': repaired,
                'alignment': alignment,
                'gold_entities': decode_entities(gold_tags),
                'predicted_entities': decode_entities(repaired),
            }
        )

    metrics = {
        'train_size': len(TRAIN_ROWS),
        'eval_size': len(EVAL_ROWS),
        'label_names': LABELS,
        'aligned_train_tokens': sum(sum(len(split_wordpieces(word)) for word in row['words']) for row in TRAIN_ROWS),
        'aligned_eval_tokens': sum(sum(len(split_wordpieces(word)) for word in row['words']) for row in EVAL_ROWS),
        'token_accuracy': rounded(token_accuracy(gold_sequences, pred_sequences)),
        **entity_metrics(gold_sequences, pred_sequences),
        'piece_majority_map': dict(sorted(piece_label_map.items())),
        'label_counts': {label: train_counts.get(label, 0) for label in LABELS},
        'alignment_example': prediction_rows[0],
        'prediction_rows': prediction_rows,
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(train_counts)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
