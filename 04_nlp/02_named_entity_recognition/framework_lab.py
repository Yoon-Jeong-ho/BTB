from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PAD_ID = 0
UNK_ID = 1
PAD_LABEL_ID = -100

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


def align_word_tags(words: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    pieces: list[str] = []
    aligned_tags: list[str] = []
    for word, tag in zip(words, tags):
        word_pieces = split_wordpieces(word)
        entity_type = tag.split('-', 1)[1] if '-' in tag else ''
        for piece_index, piece in enumerate(word_pieces):
            pieces.append(piece)
            if tag == 'O':
                aligned_tags.append('O')
            elif piece_index == 0:
                aligned_tags.append(normalize_label(tag))
            else:
                aligned_tags.append(f'I-{entity_type}' if f'I-{entity_type}' in LABELS else normalize_label(tag))
    return pieces, aligned_tags


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


def token_accuracy(gold_sequences: list[list[int]], pred_sequences: list[list[int]]) -> float:
    gold_flat = [label for sequence in gold_sequences for label in sequence]
    pred_flat = [label for sequence in pred_sequences for label in sequence]
    correct = sum(1 for gold, pred in zip(gold_flat, pred_flat) if gold == pred)
    return safe_div(correct, len(gold_flat))


def entity_metrics(gold_sequences: list[list[int]], pred_sequences: list[list[int]]) -> dict[str, float]:
    gold_entities = []
    pred_entities = []
    for index, (gold_ids, pred_ids) in enumerate(zip(gold_sequences, pred_sequences)):
        gold_entities.extend((index, *entity) for entity in decode_entities([LABELS[label_id] for label_id in gold_ids]))
        pred_entities.extend((index, *entity) for entity in decode_entities([LABELS[label_id] for label_id in pred_ids]))
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


def build_vocab(rows: list[dict[str, list[str]]]) -> dict[str, int]:
    vocab = {'[PAD]': PAD_ID, '[UNK]': UNK_ID}
    for row in rows:
        pieces, _ = align_word_tags(row['words'], row['tags'])
        for piece in pieces:
            if piece not in vocab:
                vocab[piece] = len(vocab)
    return vocab


def encode_rows(rows: list[dict[str, list[str]]], vocab: dict[str, int]) -> tuple[list[list[int]], list[list[int]], list[list[str]], list[list[str]]]:
    input_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    piece_sequences: list[list[str]] = []
    tag_sequences: list[list[str]] = []
    for row in rows:
        pieces, tags = align_word_tags(row['words'], row['tags'])
        input_sequences.append([vocab.get(piece, UNK_ID) for piece in pieces])
        label_sequences.append([LABEL_TO_ID[normalize_label(tag)] for tag in tags])
        piece_sequences.append(pieces)
        tag_sequences.append(tags)
    return input_sequences, label_sequences, piece_sequences, tag_sequences


def pad_batch(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    return torch.tensor([sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences], dtype=torch.long)


class TinySequenceLabeler(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.encoder = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = torch.nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        return self.classifier(encoded)


def run() -> None:
    torch.manual_seed(11)

    vocab = build_vocab(TRAIN_ROWS)
    train_inputs, train_labels, _, _ = encode_rows(TRAIN_ROWS, vocab)
    eval_inputs, eval_labels, eval_pieces, eval_gold_tags = encode_rows(EVAL_ROWS, vocab)

    train_input_ids = pad_batch(train_inputs, PAD_ID)
    train_label_ids = pad_batch(train_labels, PAD_LABEL_ID)
    eval_input_ids = pad_batch(eval_inputs, PAD_ID)

    model = TinySequenceLabeler(vocab_size=len(vocab), embedding_dim=20, hidden_dim=24, num_labels=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    loss_history: list[float] = []
    for _ in range(120):
        model.train()
        optimizer.zero_grad()
        logits = model(train_input_ids)
        loss = F.cross_entropy(logits.view(-1, len(LABELS)), train_label_ids.view(-1), ignore_index=PAD_LABEL_ID)
        loss.backward()
        optimizer.step()
        loss_history.append(rounded(loss.item()))

    model.eval()
    with torch.no_grad():
        eval_logits = model(eval_input_ids)
        eval_predictions = eval_logits.argmax(dim=-1).tolist()

    trimmed_predictions: list[list[int]] = []
    prediction_rows: list[dict[str, object]] = []
    for row, pieces, gold_ids, pred_ids in zip(EVAL_ROWS, eval_pieces, eval_labels, eval_predictions):
        trimmed = pred_ids[:len(gold_ids)]
        repaired_tags = repair_bio([LABELS[label_id] for label_id in trimmed])
        repaired_ids = [LABEL_TO_ID[tag] for tag in repaired_tags]
        trimmed_predictions.append(repaired_ids)
        prediction_rows.append(
            {
                'words': row['words'],
                'word_tags': row['tags'],
                'pieces': pieces,
                'gold_piece_tags': [LABELS[label_id] for label_id in gold_ids],
                'predicted_piece_tags': repaired_tags,
                'gold_entities': decode_entities([LABELS[label_id] for label_id in gold_ids]),
                'predicted_entities': decode_entities(repaired_tags),
            }
        )

    metrics = {
        'train_size': len(TRAIN_ROWS),
        'eval_size': len(EVAL_ROWS),
        'vocab_size': len(vocab),
        'embedding_dim': 20,
        'hidden_dim': 24,
        'epochs': 120,
        'num_labels': len(LABELS),
        'label_names': LABELS,
        'train_input_shape': list(train_input_ids.shape),
        'eval_input_shape': list(eval_input_ids.shape),
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'token_accuracy': rounded(token_accuracy(eval_labels, trimmed_predictions)),
        **entity_metrics(eval_labels, trimmed_predictions),
        'prediction_rows': prediction_rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
