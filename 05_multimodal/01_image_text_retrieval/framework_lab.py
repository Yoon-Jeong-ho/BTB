from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
TEMPERATURE = 0.2
EPOCHS = 200
PAIR_LABELS = [
    '주황빛 호수 카약',
    '눈 덮인 산길',
    '야간 도시 스카이라인',
    '숲속 강아지 산책',
]
TEXT_TOKENS = ['water', 'boat', 'snow', 'mountain', 'city', 'night', 'dog', 'forest', 'orange']


def build_toy_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    image_inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1, 0.9],
            [0.0, 1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0, 0.8],
            [0.2, 0.0, 0.0, 1.0, 0.7],
        ],
        dtype=torch.float32,
    )
    text_inputs = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    return image_inputs, text_inputs


def _validate_aligned_batches(image_inputs: torch.Tensor, text_inputs: torch.Tensor) -> None:
    if image_inputs.ndim != 2 or text_inputs.ndim != 2:
        raise ValueError(
            'image-text retrieval framework example expects 2D tensors shaped like '
            '(batch, dim) for both image_inputs and text_inputs.'
        )
    if image_inputs.shape[0] != text_inputs.shape[0]:
        raise ValueError(
            'image/text batch size must match for this retrieval unit toy setup: '
            f'got image batch {image_inputs.shape[0]} and text batch {text_inputs.shape[0]}.'
        )


def _rank_positions(rankings: torch.Tensor, labels: torch.Tensor) -> list[int]:
    positions: list[int] = []
    for index in range(rankings.shape[0]):
        position = (rankings[index] == labels[index]).nonzero(as_tuple=False).item() + 1
        positions.append(int(position))
    return positions


def recall_at_k(rank_positions: list[int], k: int) -> float:
    return round(float(sum(rank <= k for rank in rank_positions) / len(rank_positions)), 6)


def compute_logits(
    image_inputs: torch.Tensor,
    text_inputs: torch.Tensor,
    temperature: float,
    image_encoder: torch.nn.Module | None = None,
    text_encoder: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_aligned_batches(image_inputs, text_inputs)

    image_embeddings = image_inputs if image_encoder is None else image_encoder(image_inputs)
    text_embeddings = text_inputs if text_encoder is None else text_encoder(text_inputs)
    normalized_images = F.normalize(image_embeddings, dim=-1)
    normalized_texts = F.normalize(text_embeddings, dim=-1)
    logits = normalized_images @ normalized_texts.transpose(0, 1)
    logits = logits / temperature
    return normalized_images, normalized_texts, logits


class TinyDualEncoder(torch.nn.Module):
    def __init__(self, image_dim: int, text_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.image_encoder = torch.nn.Linear(image_dim, embedding_dim, bias=False)
        self.text_encoder = torch.nn.Linear(text_dim, embedding_dim, bias=False)

    def forward(self, image_inputs: torch.Tensor, text_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_logits(
            image_inputs=image_inputs,
            text_inputs=text_inputs,
            temperature=TEMPERATURE,
            image_encoder=self.image_encoder,
            text_encoder=self.text_encoder,
        )


def run() -> None:
    torch.manual_seed(7)
    device = torch.device('cpu')

    image_inputs, text_inputs = build_toy_inputs()
    image_inputs = image_inputs.to(device)
    text_inputs = text_inputs.to(device)
    labels = torch.arange(image_inputs.shape[0], device=device)

    model = TinyDualEncoder(image_dim=image_inputs.shape[1], text_dim=text_inputs.shape[1], embedding_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    loss_history: list[float] = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        _, _, logits = model(image_inputs, text_inputs)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.item()), 6))

    with torch.no_grad():
        normalized_images, normalized_texts, logits = model(image_inputs, text_inputs)
        similarities = normalized_images @ normalized_texts.transpose(0, 1)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
        image_to_text_rankings = similarities.argsort(dim=1, descending=True)
        text_to_image_rankings = similarities.argsort(dim=0, descending=True).transpose(0, 1)
        image_to_text_positions = _rank_positions(image_to_text_rankings, labels)
        text_to_image_positions = _rank_positions(text_to_image_rankings, labels)

    ranked_matches = []
    for index, ranking in enumerate(image_to_text_rankings.tolist()):
        ranked_matches.append(
            {
                'image_query': PAIR_LABELS[index],
                'top2_texts': [PAIR_LABELS[candidate] for candidate in ranking[:2]],
            }
        )

    metrics = {
        'device': str(device),
        'temperature': TEMPERATURE,
        'epochs': EPOCHS,
        'pair_count': int(image_inputs.shape[0]),
        'pair_labels': PAIR_LABELS,
        'text_tokens': TEXT_TOKENS,
        'image_input_shape': list(image_inputs.shape),
        'text_input_shape': list(text_inputs.shape),
        'logits_shape': list(logits.shape),
        'embedding_dim': 4,
        'loss_history_head': loss_history[:5],
        'loss_history_tail': loss_history[-5:],
        'image_to_text_ranks': image_to_text_positions,
        'text_to_image_ranks': text_to_image_positions,
        'image_to_text_recall_at_1': recall_at_k(image_to_text_positions, 1),
        'image_to_text_recall_at_2': recall_at_k(image_to_text_positions, 2),
        'text_to_image_recall_at_1': recall_at_k(text_to_image_positions, 1),
        'text_to_image_recall_at_2': recall_at_k(text_to_image_positions, 2),
        'symmetric_loss': round(float(((loss_i2t + loss_t2i) / 2.0).item()), 6),
        'max_row_probability_sum_error': round(
            float((torch.softmax(logits, dim=1).sum(dim=1) - 1.0).abs().max().item()), 8
        ),
        'logits_matrix': [
            [round(float(value), 6) for value in row]
            for row in logits.detach().cpu().tolist()
        ],
        'ranked_matches': ranked_matches,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
