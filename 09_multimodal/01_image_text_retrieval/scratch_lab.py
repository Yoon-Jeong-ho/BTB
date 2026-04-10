from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'retrieval_heatmap.svg'
TEMPERATURE = 0.25
PAIR_LABELS = [
    '주황빛 호수 카약',
    '눈 덮인 산길',
    '야간 도시 스카이라인',
    '숲속 강아지 산책',
]


def build_toy_embeddings() -> tuple[np.ndarray, np.ndarray]:
    image_embeddings = np.array(
        [
            [1.0, 0.9, 0.1, 0.0, 0.2],
            [0.0, 0.1, 1.0, 0.8, 0.0],
            [0.2, 0.0, 0.1, 1.0, 0.9],
            [0.1, 1.0, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    text_embeddings = np.array(
        [
            [0.95, 0.85, 0.2, 0.0, 0.1],
            [0.1, 0.2, 0.95, 0.75, 0.0],
            [0.15, 0.0, 0.2, 0.95, 0.85],
            [0.385, 0.7, 0.05, 0.0, 0.35],
        ],
        dtype=np.float64,
    )
    return image_embeddings, text_embeddings


def _validate_aligned_batches(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> None:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError(
            'image-text retrieval scratch example expects 2D arrays shaped like '
            '(batch, dim) for both image_embeddings and text_embeddings.'
        )
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            'image/text batch size must match for this retrieval unit toy setup: '
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


def _rank_positions(rankings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positions = []
    for index, ranking in enumerate(rankings):
        positions.append(int(np.where(ranking == labels[index])[0][0]) + 1)
    return np.array(positions, dtype=np.int64)


def recall_at_k(rank_positions: np.ndarray, k: int) -> float:
    return round(float(np.mean(rank_positions <= k)), 6)


def _hard_negative_pair(similarities: np.ndarray) -> tuple[float, tuple[int, int]]:
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    flat_index = np.argmax(np.where(mask, similarities, -np.inf))
    row, col = np.unravel_index(flat_index, similarities.shape)
    return float(similarities[row, col]), (int(row), int(col))


def _top_matches(rankings: np.ndarray) -> list[list[str]]:
    readable = []
    for ranking in rankings:
        readable.append([PAIR_LABELS[int(index)] for index in ranking[:2]])
    return readable


def retrieval_metrics(
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

    image_to_text_rankings = np.argsort(-similarities, axis=1)
    text_to_image_rankings = np.argsort(-similarities, axis=0).T
    image_to_text_positions = _rank_positions(image_to_text_rankings, labels)
    text_to_image_positions = _rank_positions(text_to_image_rankings, labels)

    diagonal = np.diag(similarities)
    off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    hardest_negative_similarity, (hard_row, hard_col) = _hard_negative_pair(similarities)
    symmetric_loss = (
        cross_entropy_from_logits(logits, labels)
        + cross_entropy_from_logits(logits.T, labels)
    ) / 2.0

    return {
        'pair_count': int(similarities.shape[0]),
        'temperature': float(temperature),
        'pair_labels': PAIR_LABELS,
        'image_embeddings_shape': list(image_embeddings.shape),
        'text_embeddings_shape': list(text_embeddings.shape),
        'similarity_matrix_shape': list(similarities.shape),
        'similarity_matrix': np.round(similarities, 6).tolist(),
        'diagonal_similarities': np.round(diagonal, 6).tolist(),
        'mean_positive_similarity': round(float(diagonal.mean()), 6),
        'mean_negative_similarity': round(float(off_diagonal.mean()), 6),
        'hardest_negative_similarity': round(float(hardest_negative_similarity), 6),
        'hardest_negative_pair': f'{PAIR_LABELS[hard_row]} ↔ {PAIR_LABELS[hard_col]}',
        'image_to_text_ranks': image_to_text_positions.astype(int).tolist(),
        'text_to_image_ranks': text_to_image_positions.astype(int).tolist(),
        'image_to_text_recall_at_1': recall_at_k(image_to_text_positions, 1),
        'image_to_text_recall_at_2': recall_at_k(image_to_text_positions, 2),
        'text_to_image_recall_at_1': recall_at_k(text_to_image_positions, 1),
        'text_to_image_recall_at_2': recall_at_k(text_to_image_positions, 2),
        'image_to_text_top2': _top_matches(image_to_text_rankings),
        'text_to_image_top2': _top_matches(text_to_image_rankings),
        'median_rank_i2t': float(np.median(image_to_text_positions)),
        'median_rank_t2i': float(np.median(text_to_image_positions)),
        'symmetric_contrastive_loss': round(float(symmetric_loss), 6),
    }


def save_heatmap_svg(similarity_matrix: np.ndarray, labels: list[str]) -> None:
    width, height = 620, 330
    margin_left, margin_top = 180, 70
    cell = 95

    def cell_color(value: float) -> str:
        normalized = (value + 1.0) / 2.0
        normalized = min(1.0, max(0.0, normalized))
        red = int(246 - 120 * normalized)
        green = int(249 - 165 * normalized)
        blue = int(255 - 185 * normalized)
        return f'#{red:02x}{green:02x}{blue:02x}'

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        '  <text x="24" y="30" font-size="22" font-family="Arial, sans-serif">Image-text retrieval heatmap</text>',
        '  <text x="24" y="52" font-size="13" font-family="Arial, sans-serif" fill="#374151">행은 image query, 열은 text candidate이며 색이 진할수록 cosine similarity가 높다.</text>',
    ]

    for idx, raw_label in enumerate(labels):
        label = escape(raw_label, quote=False)
        y = margin_top + idx * cell + cell / 2 + 5
        x = margin_left + idx * cell + cell / 2
        svg_lines.append(
            f'  <text x="24" y="{y:.1f}" font-size="13" font-family="Arial, sans-serif">{label}</text>'
        )
        svg_lines.append(
            f'  <text x="{x:.1f}" y="62" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">{label}</text>'
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
            '  <text x="24" y="310" font-size="12" font-family="Arial, sans-serif" fill="#374151">scratch에서는 텍스트 4번이 이미지 1번과도 매우 비슷하게 보여 text→image Recall@1이 0.75로 남는다.</text>',
            '</svg>',
        ]
    )
    FIGURE_PATH.write_text('\n'.join(svg_lines), encoding='utf-8')


def run() -> None:
    image_embeddings, text_embeddings = build_toy_embeddings()
    metrics = retrieval_metrics(image_embeddings, text_embeddings, temperature=TEMPERATURE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_heatmap_svg(np.array(metrics['similarity_matrix'], dtype=np.float64), PAIR_LABELS)
    metrics['figure_path'] = str(FIGURE_PATH.relative_to(UNIT_ROOT))

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
