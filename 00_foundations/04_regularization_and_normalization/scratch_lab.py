from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'training_dynamics.svg'

RAW_FEATURES = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float64)
TARGETS = (0.5 * RAW_FEATURES) + 2.0
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.1
STEPS = 6
EPSILON = 1e-9


def zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    return centered / values.std()


def _round_float(value: float) -> float:
    return round(float(value), 6)


def run_training(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    steps: int = STEPS,
) -> dict[str, object]:
    weight = 0.0
    bias = 0.0
    loss_history: list[float] = []
    grad_history: list[float] = []

    for _ in range(steps):
        predictions = (weight * features) + bias
        errors = predictions - targets
        data_loss = 0.5 * float(np.mean(errors**2))
        reg_loss = 0.5 * weight_decay * (weight**2)
        total_loss = data_loss + reg_loss

        grad_w = float(np.mean(errors * features) + (weight_decay * weight))
        grad_b = float(np.mean(errors))

        loss_history.append(total_loss)
        grad_history.append(abs(grad_w))

        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return {
        'loss_history': [_round_float(value) for value in loss_history],
        'log10_loss_history': [_round_float(math.log10(value + EPSILON)) for value in loss_history],
        'grad_history': [_round_float(value) for value in grad_history],
        'final_weight': _round_float(weight),
        'final_bias': _round_float(bias),
        'final_weight_norm': _round_float(abs(weight)),
    }


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="3" '
        f'points="{point_text}" />'
    )


def save_svg(series: dict[str, list[float]]) -> None:
    width, height = 760, 440
    left, right = 70, 690
    top, bottom = 50, 360
    x_min, x_max = 0, STEPS - 1
    all_values = [value for values in series.values() for value in values]
    y_min = min(all_values) - 0.2
    y_max = max(all_values) + 0.2

    def map_x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * (right - left)

    def map_y(value: float) -> float:
        return bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)

    colors = {
        'raw/no-reg': '#d94841',
        'normalized/no-reg': '#1c7ed6',
        'normalized+l2': '#2b8a3e',
    }
    lines = []
    legend = []
    legend_y = 74
    for label, values in series.items():
        points = [(map_x(float(step)), map_y(float(loss))) for step, loss in enumerate(values)]
        lines.append(_polyline(points, colors[label]))
        legend.append(
            f'<rect x="{right - 170}" y="{legend_y - 11}" width="12" height="12" fill="{colors[label]}" />'
            f'<text x="{right - 150}" y="{legend_y}" font-size="14" font-family="Arial, sans-serif" fill="#222">{label}</text>'
        )
        legend_y += 26

    grid_lines = []
    for step in range(STEPS):
        x = map_x(float(step))
        grid_lines.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#e9ecef" stroke-width="1" />'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="26" font-size="20" font-family="Arial, sans-serif">Training dynamics: normalization and regularization</text>
  <text x="{left}" y="44" font-size="13" font-family="Arial, sans-serif" fill="#495057">y-axis = log10(loss), same learning rate for every scenario</text>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  {''.join(lines)}
  {''.join(legend)}
  <text x="{right - 175}" y="{legend_y + 12}" font-size="12" font-family="Arial, sans-serif" fill="#666">raw input scale explodes; normalized curves stay readable</text>
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    normalized_features = zscore(RAW_FEATURES)

    raw_run = run_training(RAW_FEATURES, TARGETS, learning_rate=LEARNING_RATE)
    normalized_run = run_training(normalized_features, TARGETS, learning_rate=LEARNING_RATE)
    normalized_l2_run = run_training(
        normalized_features,
        TARGETS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(
        {
            'raw/no-reg': raw_run['log10_loss_history'],
            'normalized/no-reg': normalized_run['log10_loss_history'],
            'normalized+l2': normalized_l2_run['log10_loss_history'],
        }
    )

    metrics = {
        'raw_feature_values': [_round_float(value) for value in RAW_FEATURES],
        'normalized_feature_values': [_round_float(value) for value in normalized_features],
        'normalized_feature_mean': _round_float(float(normalized_features.mean())),
        'normalized_feature_std': _round_float(float(normalized_features.std())),
        'learning_rate': _round_float(LEARNING_RATE),
        'weight_decay': _round_float(WEIGHT_DECAY),
        'raw_initial_loss': raw_run['loss_history'][0],
        'raw_final_loss': raw_run['loss_history'][-1],
        'raw_initial_grad_norm': raw_run['grad_history'][0],
        'raw_final_grad_norm': raw_run['grad_history'][-1],
        'normalized_initial_loss': normalized_run['loss_history'][0],
        'normalized_final_loss': normalized_run['loss_history'][-1],
        'normalized_initial_grad_norm': normalized_run['grad_history'][0],
        'normalized_final_grad_norm': normalized_run['grad_history'][-1],
        'normalized_weight_norm': normalized_run['final_weight_norm'],
        'normalized_l2_final_loss': normalized_l2_run['loss_history'][-1],
        'normalized_l2_weight_norm': normalized_l2_run['final_weight_norm'],
        'raw_log10_loss_history': raw_run['log10_loss_history'],
        'normalized_log10_loss_history': normalized_run['log10_loss_history'],
        'normalized_l2_log10_loss_history': normalized_l2_run['log10_loss_history'],
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
