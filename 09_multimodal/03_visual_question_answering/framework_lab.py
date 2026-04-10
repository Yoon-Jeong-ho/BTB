from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FEATURE_NAMES = ['red', 'blue', 'ball', 'cube', 'count_one', 'count_two']
QUESTION_TOKENS = ['<pad>', 'is', 'the', 'ball', 'cube', 'red', 'what', 'color', 'how', 'many', '?']
TOKEN_TO_ID = {token: index for index, token in enumerate(QUESTION_TOKENS)}
ANSWERS = ['yes', 'no', 'red', 'blue', '1', '2']
ANSWER_TO_ID = {answer: index for index, answer in enumerate(ANSWERS)}
ANSWER_TYPES = ('yes/no', 'color', 'count')
EPOCHS = 180
LEARNING_RATE = 0.08
QUESTION_EMBED_DIM = 12
HIDDEN_DIM = 24


def build_toy_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, str]], list[str]]:
    image_inputs = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    question_tokens = torch.tensor(
        [
            [TOKEN_TO_ID['is'], TOKEN_TO_ID['the'], TOKEN_TO_ID['ball'], TOKEN_TO_ID['red'], TOKEN_TO_ID['?']],
            [TOKEN_TO_ID['is'], TOKEN_TO_ID['the'], TOKEN_TO_ID['ball'], TOKEN_TO_ID['red'], TOKEN_TO_ID['?']],
            [TOKEN_TO_ID['what'], TOKEN_TO_ID['color'], TOKEN_TO_ID['cube'], TOKEN_TO_ID['?'], TOKEN_TO_ID['<pad>']],
            [TOKEN_TO_ID['how'], TOKEN_TO_ID['many'], TOKEN_TO_ID['cube'], TOKEN_TO_ID['?'], TOKEN_TO_ID['<pad>']],
            [TOKEN_TO_ID['how'], TOKEN_TO_ID['many'], TOKEN_TO_ID['cube'], TOKEN_TO_ID['?'], TOKEN_TO_ID['<pad>']],
            [TOKEN_TO_ID['what'], TOKEN_TO_ID['color'], TOKEN_TO_ID['ball'], TOKEN_TO_ID['?'], TOKEN_TO_ID['<pad>']],
        ],
        dtype=torch.long,
    )
    answer_ids = torch.tensor(
        [ANSWER_TO_ID['yes'], ANSWER_TO_ID['no'], ANSWER_TO_ID['blue'], ANSWER_TO_ID['2'], ANSWER_TO_ID['1'], ANSWER_TO_ID['blue']],
        dtype=torch.long,
    )
    rows = [
        {'image_label': '빨간 공 한 개', 'question': '공은 빨간색인가?', 'answer_type': 'yes/no', 'gold_answer': 'yes'},
        {'image_label': '파란 공 한 개', 'question': '공은 빨간색인가?', 'answer_type': 'yes/no', 'gold_answer': 'no'},
        {'image_label': '파란 큐브 한 개', 'question': '큐브 색은 무엇인가?', 'answer_type': 'color', 'gold_answer': 'blue'},
        {'image_label': '빨간 큐브 두 개', 'question': '큐브는 몇 개인가?', 'answer_type': 'count', 'gold_answer': '2'},
        {'image_label': '빨간 큐브 한 개', 'question': '큐브는 몇 개인가?', 'answer_type': 'count', 'gold_answer': '1'},
        {'image_label': '파란 공 두 개', 'question': '공 색은 무엇인가?', 'answer_type': 'color', 'gold_answer': 'blue'},
    ]
    return image_inputs, question_tokens, answer_ids, rows, FEATURE_NAMES


def _validate_inputs(image_inputs: torch.Tensor, question_tokens: torch.Tensor) -> None:
    if image_inputs.ndim != 2 or question_tokens.ndim != 2:
        raise ValueError(
            'framework visual question answering example expects 2D tensors shaped like '
            '(batch, feature_dim) for image_inputs and (batch, steps) for question_tokens.'
        )
    if image_inputs.shape[0] != question_tokens.shape[0]:
        raise ValueError(
            'image/question batch size must match for this visual question answering toy setup: '
            f'got image batch {image_inputs.shape[0]} and question batch {question_tokens.shape[0]}.'
        )


class TinyVQAModel(torch.nn.Module):
    def __init__(self, image_dim: int, question_vocab_size: int, embed_dim: int, hidden_dim: int, answer_vocab_size: int) -> None:
        super().__init__()
        self.question_embedding = torch.nn.Embedding(question_vocab_size, embed_dim, padding_idx=TOKEN_TO_ID['<pad>'])
        self.image_projection = torch.nn.Linear(image_dim, hidden_dim)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + embed_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, answer_vocab_size),
        )

    def forward(self, image_inputs: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        _validate_inputs(image_inputs, question_tokens)
        token_embeddings = self.question_embedding(question_tokens)
        mask = (question_tokens != TOKEN_TO_ID['<pad>']).unsqueeze(-1)
        token_sums = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp_min(1)
        question_repr = token_sums / token_counts
        image_repr = self.image_projection(image_inputs)
        fused = torch.cat([image_repr, question_repr], dim=-1)
        return self.fusion(fused)


def compute_vqa_logits(
    image_inputs: torch.Tensor,
    question_tokens: torch.Tensor,
    model: TinyVQAModel | None = None,
) -> torch.Tensor:
    _validate_inputs(image_inputs, question_tokens)
    if model is None:
        model = TinyVQAModel(
            image_dim=image_inputs.shape[1],
            question_vocab_size=len(QUESTION_TOKENS),
            embed_dim=QUESTION_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            answer_vocab_size=len(ANSWERS),
        )
    return model(image_inputs, question_tokens)


def _answer_type_accuracy(rows: list[dict[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for answer_type in ANSWER_TYPES:
        subset = [row for row in rows if row['answer_type'] == answer_type]
        if not subset:
            raise ValueError(f'Missing answer_type bucket for VQA accuracy: {answer_type}')
        correct = sum(bool(row['is_correct']) for row in subset)
        result[answer_type] = round(float(correct / len(subset)), 6)
    return result


def run() -> None:
    torch.manual_seed(11)
    device = torch.device('cpu')

    image_inputs, question_tokens, answer_ids, row_meta, _ = build_toy_dataset()
    image_inputs = image_inputs.to(device)
    question_tokens = question_tokens.to(device)
    answer_ids = answer_ids.to(device)

    model = TinyVQAModel(
        image_dim=image_inputs.shape[1],
        question_vocab_size=len(QUESTION_TOKENS),
        embed_dim=QUESTION_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        answer_vocab_size=len(ANSWERS),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history: list[float] = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(image_inputs, question_tokens)
        loss = F.cross_entropy(logits, answer_ids)
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.item()), 6))

    with torch.no_grad():
        logits = model(image_inputs, question_tokens)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    rows: list[dict[str, object]] = []
    for meta, predicted_id, answer_id, probability_row in zip(row_meta, predictions.tolist(), answer_ids.tolist(), probabilities.tolist()):
        predicted_answer = ANSWERS[predicted_id]
        gold_answer = ANSWERS[answer_id]
        rows.append(
            {
                'image_label': meta['image_label'],
                'question': meta['question'],
                'answer_type': meta['answer_type'],
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'is_correct': predicted_answer == gold_answer,
                'confidence': round(float(max(probability_row)), 6),
            }
        )

    overall_accuracy = round(float((predictions == answer_ids).float().mean().item()), 6)
    metrics = {
        'device': str(device),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'image_input_shape': list(image_inputs.shape),
        'question_token_shape': list(question_tokens.shape),
        'question_vocab_size': len(QUESTION_TOKENS),
        'answer_vocab_size': len(ANSWERS),
        'hidden_dim': HIDDEN_DIM,
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'question_accuracy': overall_accuracy,
        'overall_accuracy': overall_accuracy,
        'answer_type_accuracy': _answer_type_accuracy(rows),
        'rows': rows,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
