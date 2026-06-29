from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'activation_curves.svg'


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive_mask = values >= 0
    negative_mask = ~positive_mask
    result = np.empty_like(values, dtype=np.float64)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[negative_mask])
    result[negative_mask] = exp_values / (1.0 + exp_values)
    return result


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def binary_cross_entropy_from_logit(logit: float, target: float) -> tuple[float, float]:
    if logit >= 0.0:
        exp_term = math.exp(-logit)
        probability = 1.0 / (1.0 + exp_term)
    else:
        exp_term = math.exp(logit)
        probability = exp_term / (1.0 + exp_term)

    loss = max(logit, 0.0) - (logit * target) + math.log1p(math.exp(-abs(logit)))
    return probability, loss


def cross_entropy_from_logits(logits: np.ndarray, target_index: int) -> tuple[np.ndarray, float]:
    probabilities = softmax(logits, axis=-1)
    loss = -math.log(float(probabilities[target_index]))
    return probabilities, loss


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'points="{point_text}" />'
    )


def save_svg(
    x_values: np.ndarray,
    relu_values: np.ndarray,
    sigmoid_values: np.ndarray,
    tanh_values: np.ndarray,
) -> None:
    width, height = 900, 420
    top, bottom = 50, 360
    full_left, full_right = 64, 430
    zoom_left, zoom_right = 520, 805
    x_min, x_max = -4.5, 4.5
    full_y_min, full_y_max = -1.2, 4.5
    zoom_y_min, zoom_y_max = -1.05, 1.05

    def map_x(value: float, left: float, right: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def map_y(value: float, y_min: float, y_max: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    full_relu_points = [
        (map_x(float(x), full_left, full_right), map_y(float(y), full_y_min, full_y_max))
        for x, y in zip(x_values, relu_values)
    ]
    full_sigmoid_points = [
        (map_x(float(x), full_left, full_right), map_y(float(y), full_y_min, full_y_max))
        for x, y in zip(x_values, sigmoid_values)
    ]
    full_tanh_points = [
        (map_x(float(x), full_left, full_right), map_y(float(y), full_y_min, full_y_max))
        for x, y in zip(x_values, tanh_values)
    ]
    zoom_sigmoid_points = [
        (map_x(float(x), zoom_left, zoom_right), map_y(float(y), zoom_y_min, zoom_y_max))
        for x, y in zip(x_values, sigmoid_values)
    ]
    zoom_tanh_points = [
        (map_x(float(x), zoom_left, zoom_right), map_y(float(y), zoom_y_min, zoom_y_max))
        for x, y in zip(x_values, tanh_values)
    ]
    sigmoid_mid_x = map_x(0.0, zoom_left, zoom_right)
    sigmoid_mid_y = map_y(0.5, zoom_y_min, zoom_y_max)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{full_left}" y="24" font-size="20" font-family="Arial, sans-serif">Activation curves (scratch)</text>
  <text x="{full_left}" y="44" font-size="12" font-family="Arial, sans-serif" fill="#666">left: shared scale with ReLU, right: bounded activation zoom, dense 161-point curve</text>

  <text x="{full_left}" y="390" font-size="12" font-family="Arial, sans-serif" fill="#555">shared y-scale: ReLU makes sigmoid look vertically compressed</text>
  <line x1="{full_left}" y1="{bottom}" x2="{full_right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{map_x(0, full_left, full_right)}" y1="{top}" x2="{map_x(0, full_left, full_right)}" y2="{bottom}" stroke="#999" stroke-width="1.5" stroke-dasharray="4 4" />
  <line x1="{full_left}" y1="{map_y(0, full_y_min, full_y_max)}" x2="{full_right}" y2="{map_y(0, full_y_min, full_y_max)}" stroke="#999" stroke-width="1.5" stroke-dasharray="4 4" />
  {_polyline(full_relu_points, '#d94841')}
  {_polyline(full_sigmoid_points, '#2b8a3e')}
  {_polyline(full_tanh_points, '#1c7ed6')}

  <text x="{zoom_left}" y="390" font-size="12" font-family="Arial, sans-serif" fill="#555">zoom y-scale [-1, 1]: sigmoid S-curve and tanh saturation are visible</text>
  <line x1="{zoom_left}" y1="{bottom}" x2="{zoom_right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{map_x(0, zoom_left, zoom_right)}" y1="{top}" x2="{map_x(0, zoom_left, zoom_right)}" y2="{bottom}" stroke="#999" stroke-width="1.5" stroke-dasharray="4 4" />
  <line x1="{zoom_left}" y1="{map_y(0, zoom_y_min, zoom_y_max)}" x2="{zoom_right}" y2="{map_y(0, zoom_y_min, zoom_y_max)}" stroke="#999" stroke-width="1.5" stroke-dasharray="4 4" />
  {_polyline(zoom_sigmoid_points, '#2b8a3e')}
  {_polyline(zoom_tanh_points, '#1c7ed6')}
  <circle cx="{sigmoid_mid_x}" cy="{sigmoid_mid_y}" r="4" fill="#2b8a3e" />
  <text x="{sigmoid_mid_x + 8}" y="{sigmoid_mid_y - 8}" font-size="12" font-family="Arial, sans-serif" fill="#2b8a3e">sigmoid(0)=0.5</text>

  <text x="830" y="70" font-size="14" font-family="Arial, sans-serif" fill="#d94841">ReLU</text>
  <text x="830" y="94" font-size="14" font-family="Arial, sans-serif" fill="#2b8a3e">sigmoid</text>
  <text x="830" y="118" font-size="14" font-family="Arial, sans-serif" fill="#1c7ed6">tanh</text>
  <rect x="810" y="54" width="12" height="12" fill="#d94841" />
  <rect x="810" y="78" width="12" height="12" fill="#2b8a3e" />
  <rect x="810" y="102" width="12" height="12" fill="#1c7ed6" />
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    x_values = np.linspace(-4.0, 4.0, 9)
    relu_values = np.maximum(0.0, x_values)
    sigmoid_values = sigmoid(x_values)
    tanh_values = np.tanh(x_values)
    plot_x_values = np.linspace(-4.0, 4.0, 161)
    plot_relu_values = np.maximum(0.0, plot_x_values)
    plot_sigmoid_values = sigmoid(plot_x_values)
    plot_tanh_values = np.tanh(plot_x_values)

    class_logits = np.array([2.2, 0.3, -1.4], dtype=np.float64)
    class_probabilities, cross_entropy = cross_entropy_from_logits(class_logits, target_index=0)
    binary_probability, binary_cross_entropy = binary_cross_entropy_from_logit(1.25, target=1.0)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(plot_x_values, plot_relu_values, plot_sigmoid_values, plot_tanh_values)

    metrics = {
        'x_values': [round(float(value), 6) for value in x_values],
        'figure_sample_count': int(plot_x_values.size),
        'relu_values': [round(float(value), 6) for value in relu_values],
        'sigmoid_values': [round(float(value), 6) for value in sigmoid_values],
        'tanh_values': [round(float(value), 6) for value in tanh_values],
        'relu_zero_fraction': round(float((relu_values == 0.0).mean()), 6),
        'sigmoid_midpoint': round(float(sigmoid(np.array([0.0]))[0]), 6),
        'tanh_midpoint': round(float(np.tanh(0.0)), 6),
        'softmax_input_logits': [round(float(value), 6) for value in class_logits],
        'softmax_probabilities': [round(float(value), 6) for value in class_probabilities],
        'softmax_probability_sum': round(float(class_probabilities.sum()), 6),
        'softmax_argmax': int(class_probabilities.argmax()),
        'binary_probability': round(binary_probability, 6),
        'binary_cross_entropy': round(binary_cross_entropy, 6),
        'cross_entropy': round(cross_entropy, 6),
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
