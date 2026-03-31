from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
LABELS = ['negative', 'positive']
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PAD_ID = 0
UNK_ID = 1
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


def accuracy_score(gold: list[int], pred: list[int]) -> float:
    matches = sum(1 for expected, actual in zip(gold, pred) if expected == actual)
    return _safe_div(matches, len(gold))


def macro_f1_score(gold: list[int], pred: list[int]) -> float:
    f1_values: list[float] = []
    for label_id in range(len(LABELS)):
        tp = sum(1 for expected, actual in zip(gold, pred) if expected == label_id and actual == label_id)
        fp = sum(1 for expected, actual in zip(gold, pred) if expected != label_id and actual == label_id)
        fn = sum(1 for expected, actual in zip(gold, pred) if expected == label_id and actual != label_id)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1_values.append(_safe_div(2 * precision * recall, precision + recall))
    return _safe_div(sum(f1_values), len(f1_values))


def _rounded(value: float) -> float:
    return round(float(value), 6)


def build_vocab(rows: list[tuple[str, str]]) -> dict[str, int]:
    vocab = {'[PAD]': PAD_ID, '[UNK]': UNK_ID}
    for text, _ in rows:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(token, UNK_ID) for token in tokenize(text)]


def pad_batch(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)

    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [PAD_ID] * (max_len - len(sequence)) for sequence in sequences]
    input_ids = torch.tensor(padded, dtype=torch.long)
    mask = input_ids.ne(PAD_ID).float()
    return input_ids, mask


class TinyTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        masked = embedded * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.classifier(pooled)


def run() -> None:
    torch.manual_seed(7)

    vocab = build_vocab(TRAIN_ROWS)
    train_sequences = [encode_text(text, vocab) for text, _ in TRAIN_ROWS]
    train_labels = torch.tensor([LABEL_TO_ID[label] for _, label in TRAIN_ROWS], dtype=torch.long)
    eval_sequences = [encode_text(text, vocab) for text, _ in EVAL_ROWS]
    eval_labels = [LABEL_TO_ID[label] for _, label in EVAL_ROWS]

    train_input_ids, train_mask = pad_batch(train_sequences)
    eval_input_ids, eval_mask = pad_batch(eval_sequences)

    model = TinyTextClassifier(vocab_size=len(vocab), embedding_dim=12, num_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    loss_history: list[float] = []
    for _ in range(80):
        model.train()
        optimizer.zero_grad()
        logits = model(train_input_ids, train_mask)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        loss_history.append(_rounded(loss.item()))

    model.eval()
    with torch.no_grad():
        eval_logits = model(eval_input_ids, eval_mask)
        eval_probabilities = torch.softmax(eval_logits, dim=-1)
        eval_predictions = eval_logits.argmax(dim=-1).tolist()

    prediction_rows = []
    for (text, gold_label), probabilities, predicted_id in zip(EVAL_ROWS, eval_probabilities.tolist(), eval_predictions):
        prediction_rows.append(
            {
                'text': text,
                'gold': gold_label,
                'predicted': LABELS[predicted_id],
                'probabilities': {label: _rounded(prob) for label, prob in zip(LABELS, probabilities)},
            }
        )

    metrics = {
        'train_size': len(TRAIN_ROWS),
        'eval_size': len(EVAL_ROWS),
        'vocab_size': len(vocab),
        'embedding_dim': 12,
        'epochs': 80,
        'num_classes': len(LABELS),
        'label_names': LABELS,
        'train_input_shape': list(train_input_ids.shape),
        'eval_input_shape': list(eval_input_ids.shape),
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'eval_accuracy': _rounded(accuracy_score(eval_labels, eval_predictions)),
        'eval_macro_f1': _rounded(macro_f1_score(eval_labels, eval_predictions)),
        'prediction_rows': prediction_rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
