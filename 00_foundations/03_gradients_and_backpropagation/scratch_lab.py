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


def _polyline(points: list[tuple[float, float]], color: str, *, width: int = 3) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{point_text}" />'
    )


def save_svg(current_weight: float, updated_weight: float, bias: float, updated_bias: float) -> None:
    width, height = 820, 470
    left, right = 78, 560
    top, bottom = 78, 365
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
    updated_loss_on_weight_slice = forward_loss(updated_weight, bias)[1]
    _, actual_updated_loss = forward_loss(updated_weight, updated_bias)

    grid_lines = []
    for loss_tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = map_y(loss_tick)
        grid_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#edf2f7" />')
        grid_lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#495057">{loss_tick:.2f}</text>')
    x_ticks = []
    for weight_tick in [0.2, 0.5, 0.8, 1.1, 1.4]:
        x = map_x(weight_tick)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}" stroke="#222" />')
        x_ticks.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#495057">{weight_tick:.1f}</text>')

    current_x, current_y = map_x(current_weight), map_y(current_loss)
    updated_x, updated_y = map_x(updated_weight), map_y(updated_loss_on_weight_slice)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="42" y="32" font-size="21" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Loss curve around w (scratch backprop)</text>
  <text x="42" y="54" font-size="13" font-family="Arial, sans-serif" fill="#495057">The red point is the current parameter; the green point shows one gradient step moving toward lower loss.</text>
  <rect x="42" y="68" width="746" height="360" rx="18" fill="#f8fafc" stroke="#e9ecef" />
  {''.join(grid_lines)}
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  {''.join(x_ticks)}
  <text x="{(left + right) / 2 - 35:.2f}" y="{bottom + 48}" font-size="12" font-family="Arial, sans-serif" fill="#495057">weight w</text>
  <text transform="rotate(-90 {left - 54:.2f} {(top + bottom) / 2:.2f})" x="{left - 54:.2f}" y="{(top + bottom) / 2:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#495057">loss</text>
  {_polyline(curve_points, '#1c7ed6')}
  <line x1="{current_x:.2f}" y1="{current_y:.2f}" x2="{updated_x:.2f}" y2="{updated_y:.2f}" stroke="#495057" stroke-width="2" stroke-dasharray="5 4" />
  <circle cx="{current_x:.2f}" cy="{current_y:.2f}" r="7" fill="#d94841" stroke="#ffffff" stroke-width="2" />
  <circle cx="{updated_x:.2f}" cy="{updated_y:.2f}" r="7" fill="#2b8a3e" stroke="#ffffff" stroke-width="2" />
  <text x="{current_x + 10:.2f}" y="{current_y - 12:.2f}" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#d94841">loss before update: {current_loss:.3f}</text>
  <text x="{updated_x - 126:.2f}" y="{updated_y - 10:.2f}" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#2b8a3e">lower after one step</text>
  <rect x="590" y="105" width="170" height="180" rx="14" fill="#ffffff" stroke="#d0d7de" />
  <text x="606" y="132" font-size="14" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Read this figure</text>
  <text x="606" y="160" font-size="12" font-family="Arial, sans-serif" fill="#495057">1. Find the red point.</text>
  <text x="606" y="184" font-size="12" font-family="Arial, sans-serif" fill="#495057">2. Follow the dashed move.</text>
  <text x="606" y="208" font-size="12" font-family="Arial, sans-serif" fill="#495057">3. Green is lower loss.</text>
  <text x="606" y="240" font-size="12" font-family="Arial, sans-serif" fill="#212529">Actual full update:</text>
  <text x="606" y="264" font-size="12" font-family="Arial, sans-serif" fill="#212529">{current_loss:.3f} -> {actual_updated_loss:.3f}</text>
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
    save_svg(current_weight=WEIGHT, updated_weight=updated_weight, bias=BIAS, updated_bias=updated_bias)

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
