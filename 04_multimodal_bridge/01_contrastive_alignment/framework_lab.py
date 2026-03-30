from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
TEMPERATURE = 0.2
PAIR_LABELS = ['붉은 원 ↔ red circle', '푸른 사각형 ↔ blue square', '초록 삼각형 ↔ green triangle']


def build_toy_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.1, 1.0, 0.0],
            [0.0, 0.2, 1.0],
        ],
        dtype=torch.float32,
    )
    text_embeddings = torch.tensor(
        [
            [0.95, 0.05, 0.1],
            [0.1, 0.92, 0.05],
            [0.05, 0.12, 0.96],
        ],
        dtype=torch.float32,
    )
    return image_embeddings, text_embeddings


def _validate_aligned_batches(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> None:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError(
            'contrastive alignment toy example expects 2D tensors shaped like '
            '(batch, dim) for both image_embeddings and text_embeddings.'
        )
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            'image/text batch size must match for this bridge unit toy alignment: '
            f'got image batch {image_embeddings.shape[0]} and text batch {text_embeddings.shape[0]}.'
        )


def compute_logits(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_aligned_batches(image_embeddings, text_embeddings)

    normalized_images = F.normalize(image_embeddings, dim=-1)
    normalized_texts = F.normalize(text_embeddings, dim=-1)
    logits = normalized_images @ normalized_texts.transpose(0, 1)
    logits = logits / temperature
    return normalized_images, normalized_texts, logits


def run() -> None:
    torch.manual_seed(0)
    device = torch.device('cpu')

    image_embeddings, text_embeddings = build_toy_embeddings()
    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    normalized_images, normalized_texts, logits = compute_logits(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        temperature=TEMPERATURE,
    )

    labels = torch.arange(logits.shape[0], device=device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
    predictions = logits.argmax(dim=1)

    metrics = {
        'device': str(device),
        'temperature': TEMPERATURE,
        'pair_labels': PAIR_LABELS,
        'image_embeddings_shape': list(image_embeddings.shape),
        'text_embeddings_shape': list(text_embeddings.shape),
        'normalized_image_shape': list(normalized_images.shape),
        'normalized_text_shape': list(normalized_texts.shape),
        'logits_shape': list(logits.shape),
        'labels_shape': list(labels.shape),
        'dtype': str(logits.dtype).replace('torch.', ''),
        'logits_matrix': [
            [round(float(value), 6) for value in row]
            for row in logits.detach().cpu().tolist()
        ],
        'diagonal_logits': [
            round(float(value), 6) for value in torch.diagonal(logits).detach().cpu().tolist()
        ],
        'top1_predictions': [int(value) for value in predictions.detach().cpu().tolist()],
        'top1_alignment_accuracy': round(float((predictions == labels).float().mean().item()), 6),
        'loss_i2t': round(float(loss_i2t.item()), 6),
        'loss_t2i': round(float(loss_t2i.item()), 6),
        'symmetric_loss': round(float(((loss_i2t + loss_t2i) / 2.0).item()), 6),
        'max_row_probability_sum_error': round(
            float((torch.softmax(logits, dim=1).sum(dim=1) - 1.0).abs().max().item()), 8
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
