from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'loss_curve.svg'


X_VALUE = 1.5
WEIGHT = 0.8
BIAS = -0.4
TARGET = 0.3
LEARNING_RATE = 0.1
EPSILON = 1e-5


def forward_loss(weight: float, bias: float, x_value: float = X_VALUE, target: float = TARGET) -> tuple[float, float]:
    prediction = (weight * x_value) + bias
    error = prediction - target
    loss = 0.5 * (error**2)
    return prediction, loss


def analytic_gradients(weight: float, bias: float, x_value: float = X_VALUE, target: float = TARGET) -> tuple[float, float, float, float]:
    prediction, loss = forward_loss(weight=weight, bias=bias, x_value=x_value, target=target)
    dloss_dprediction = prediction - target
    grad_w = dloss_dprediction * x_value
    grad_b = dloss_dprediction
    return prediction, loss, grad_w, grad_b


def finite_difference_weight(weight: float, bias: float, epsilon: float = EPSILON) -> float:
    _, loss_plus = forward_loss(weight + epsilon, bias)
    _, loss_minus = forward_loss(weight - epsilon, bias)
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def finite_difference_bias(weight: float, bias: float, epsilon: float = EPSILON) -> float:
    _, loss_plus = forward_loss(weight, bias + epsilon)
    _, loss_minus = forward_loss(weight, bias - epsilon)
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'points="{point_text}" />'
    )


def save_svg(current_weight: float, updated_weight: float, bias: float) -> None:
    width, height = 720, 420
    left, right = 70, 660
    top, bottom = 40, 360
    weight_min, weight_max = 0.2, 1.4

    sampled_weights = [weight_min + index * 0.1 for index in range(13)]
    sampled_losses = [forward_loss(weight=value, bias=bias)[1] for value in sampled_weights]
    loss_min = 0.0
    loss_max = max(sampled_losses) * 1.1

    def map_x(value: float) -> float:
        return left + (value - weight_min) / (weight_max - weight_min) * (right - left)

    def map_y(value: float) -> float:
        return bottom - (value - loss_min) / (loss_max - loss_min) * (bottom - top)

    curve_points = [(map_x(weight), map_y(loss)) for weight, loss in zip(sampled_weights, sampled_losses)]
    current_loss = forward_loss(current_weight, bias)[1]
    updated_loss = forward_loss(updated_weight, bias)[1]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="24" font-size="20" font-family="Arial, sans-serif">Loss curve around w (scratch backprop)</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  {_polyline(curve_points, '#1c7ed6')}
  <circle cx="{map_x(current_weight)}" cy="{map_y(current_loss)}" r="6" fill="#d94841" />
  <circle cx="{map_x(updated_weight)}" cy="{map_y(updated_loss)}" r="6" fill="#2b8a3e" />
  <text x="{right - 150}" y="70" font-size="14" font-family="Arial, sans-serif" fill="#d94841">current w</text>
  <text x="{right - 150}" y="95" font-size="14" font-family="Arial, sans-serif" fill="#2b8a3e">updated w</text>
  <rect x="{right - 170}" y="58" width="12" height="12" fill="#d94841" />
  <rect x="{right - 170}" y="83" width="12" height="12" fill="#2b8a3e" />
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    prediction, loss, grad_w, grad_b = analytic_gradients(WEIGHT, BIAS)
    fd_grad_w = finite_difference_weight(WEIGHT, BIAS)
    fd_grad_b = finite_difference_bias(WEIGHT, BIAS)

    updated_weight = WEIGHT - (LEARNING_RATE * grad_w)
    updated_bias = BIAS - (LEARNING_RATE * grad_b)
    updated_prediction, updated_loss = forward_loss(updated_weight, updated_bias)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(current_weight=WEIGHT, updated_weight=updated_weight, bias=BIAS)

    metrics = {
        'x_value': round(X_VALUE, 6),
        'weight': round(WEIGHT, 6),
        'bias': round(BIAS, 6),
        'target': round(TARGET, 6),
        'prediction': round(prediction, 6),
        'loss': round(loss, 6),
        'dloss_dprediction': round(prediction - TARGET, 6),
        'grad_w': round(grad_w, 6),
        'grad_b': round(grad_b, 6),
        'finite_diff_grad_w': round(fd_grad_w, 6),
        'finite_diff_grad_b': round(fd_grad_b, 6),
        'grad_error_w': round(abs(grad_w - fd_grad_w), 12),
        'grad_error_b': round(abs(grad_b - fd_grad_b), 12),
        'learning_rate': round(LEARNING_RATE, 6),
        'updated_weight': round(updated_weight, 6),
        'updated_bias': round(updated_bias, 6),
        'updated_prediction': round(updated_prediction, 6),
        'updated_loss': round(updated_loss, 6),
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
