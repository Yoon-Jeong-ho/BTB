from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'alignment_heatmap.svg'
TEMPERATURE = 0.2
PAIR_LABELS = ['붉은 원 ↔ red circle', '푸른 사각형 ↔ blue square', '초록 삼각형 ↔ green triangle']


def build_toy_embeddings() -> tuple[np.ndarray, np.ndarray]:
    image_embeddings = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.1, 1.0, 0.0],
            [0.0, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    text_embeddings = np.array(
        [
            [0.95, 0.05, 0.1],
            [0.1, 0.92, 0.05],
            [0.05, 0.12, 0.96],
        ],
        dtype=np.float64,
    )
    return image_embeddings, text_embeddings


def _validate_aligned_batches(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> None:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError(
            'contrastive alignment toy example expects 2D arrays shaped like '
            '(batch, dim) for both image_embeddings and text_embeddings.'
        )
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            'image/text batch size must match for this bridge unit toy alignment: '
            f'got image batch {image_embeddings.shape[0]} and text batch {text_embeddings.shape[0]}.'
        )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = softmax(logits, axis=-1)
    picked = probabilities[np.arange(labels.shape[0]), labels]
    return float(np.mean(-np.log(np.clip(picked, 1e-12, 1.0))))


def contrastive_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float,
) -> dict[str, object]:
    _validate_aligned_batches(image_embeddings, text_embeddings)

    normalized_images = l2_normalize(image_embeddings)
    normalized_texts = l2_normalize(text_embeddings)
    similarities = normalized_images @ normalized_texts.T
    logits = similarities / temperature
    labels = np.arange(similarities.shape[0])

    diagonal = np.diag(similarities)
    off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    loss_i2t = cross_entropy_from_logits(logits, labels)
    loss_t2i = cross_entropy_from_logits(logits.T, labels)
    predictions = np.argmax(similarities, axis=1)

    return {
        'pair_count': int(similarities.shape[0]),
        'temperature': float(temperature),
        'image_embeddings_shape': list(image_embeddings.shape),
        'text_embeddings_shape': list(text_embeddings.shape),
        'similarity_matrix_shape': list(similarities.shape),
        'pair_labels': PAIR_LABELS,
        'similarity_matrix': np.round(similarities, 6).tolist(),
        'logits_matrix': np.round(logits, 6).tolist(),
        'diagonal_similarities': np.round(diagonal, 6).tolist(),
        'mean_positive_similarity': round(float(diagonal.mean()), 6),
        'mean_negative_similarity': round(float(off_diagonal.mean()), 6),
        'hardest_negative_similarity': round(float(off_diagonal.max()), 6),
        'top1_predictions': predictions.astype(int).tolist(),
        'top1_alignment_accuracy': round(float(np.mean(predictions == labels)), 6),
        'loss_i2t': round(loss_i2t, 6),
        'loss_t2i': round(loss_t2i, 6),
        'symmetric_contrastive_loss': round((loss_i2t + loss_t2i) / 2.0, 6),
    }


def save_heatmap_svg(similarity_matrix: np.ndarray, labels: list[str]) -> None:
    width, height = 560, 280
    margin_left, margin_top = 170, 60
    cell = 90

    def cell_color(value: float) -> str:
        normalized = (value + 1.0) / 2.0
        normalized = min(1.0, max(0.0, normalized))
        red = int(245 - 120 * normalized)
        green = int(248 - 170 * normalized)
        blue = int(255 - 190 * normalized)
        return f'#{red:02x}{green:02x}{blue:02x}'

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        '  <text x="24" y="28" font-size="20" font-family="Arial, sans-serif">Contrastive alignment heatmap</text>',
    ]

    for idx, raw_label in enumerate(labels):
        label = escape(raw_label, quote=False)
        y = margin_top + idx * cell + cell / 2 + 5
        x = margin_left + idx * cell + cell / 2
        svg_lines.append(
            f'  <text x="20" y="{y:.1f}" font-size="13" font-family="Arial, sans-serif">{label}</text>'
        )
        svg_lines.append(
            f'  <text x="{x:.1f}" y="48" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{label}</text>'
        )

    for row in range(similarity_matrix.shape[0]):
        for col in range(similarity_matrix.shape[1]):
            value = float(similarity_matrix[row, col])
            x = margin_left + col * cell
            y = margin_top + row * cell
            stroke = '#1f2937' if row == col else '#94a3b8'
            stroke_width = 3 if row == col else 1.5
            svg_lines.extend(
                [
                    f'  <rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{cell_color(value)}" stroke="{stroke}" stroke-width="{stroke_width}" />',
                    f'  <text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 5:.1f}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111827">{value:.3f}</text>',
                ]
            )

    svg_lines.extend(
        [
            '  <text x="24" y="255" font-size="12" font-family="Arial, sans-serif" fill="#374151">대각선이 정답 이미지-텍스트 쌍이며, 색이 진할수록 cosine similarity가 높다.</text>',
            '</svg>',
        ]
    )
    FIGURE_PATH.write_text('\n'.join(svg_lines), encoding='utf-8')


def run() -> None:
    image_embeddings, text_embeddings = build_toy_embeddings()
    metrics = contrastive_metrics(image_embeddings, text_embeddings, temperature=TEMPERATURE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_heatmap_svg(np.array(metrics['similarity_matrix'], dtype=np.float64), PAIR_LABELS)
    metrics['figure_path'] = str(FIGURE_PATH.relative_to(UNIT_ROOT))

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
