from __future__ import annotations

import itertools
import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'decision_regions.svg'

LINEAR_POINTS = [(-2.0, -1.0), (-1.0, -2.0), (1.0, 1.0), (2.0, 1.0)]
LINEAR_LABELS = [0, 0, 1, 1]
XOR_POINTS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
XOR_LABELS = [0, 1, 1, 0]
GRID_VALUES = (-2, -1, 0, 1, 2)
LINEAR_RULE = {'weights': (1.0, 0.0), 'bias': 0.0}
XOR_RULE = {'weights': (1.0, 1.0), 'bias': 0.0}


def predict_label(point: tuple[float, float], weights: tuple[float, float], bias: float) -> tuple[int, float]:
    score = (weights[0] * point[0]) + (weights[1] * point[1]) + bias
    return int(score >= 0.0), score


def evaluate_dataset(
    points: list[tuple[float, float]],
    labels: list[int],
    weights: tuple[float, float],
    bias: float,
) -> dict[str, object]:
    predictions: list[int] = []
    scores: list[float] = []
    for point in points:
        prediction, score = predict_label(point, weights, bias)
        predictions.append(prediction)
        scores.append(score)
    correct = sum(int(pred == gold) for pred, gold in zip(predictions, labels))
    accuracy = correct / len(labels)
    return {
        'weights': [float(weights[0]), float(weights[1])],
        'bias': float(bias),
        'predictions': predictions,
        'scores': [round(score, 6) for score in scores],
        'accuracy': round(accuracy, 6),
    }


def best_single_neuron_accuracy(points: list[tuple[float, float]], labels: list[int]) -> float:
    best_accuracy = 0.0
    for weight_x, weight_y, bias in itertools.product(GRID_VALUES, repeat=3):
        if weight_x == 0 and weight_y == 0:
            continue
        accuracy = evaluate_dataset(points, labels, (float(weight_x), float(weight_y)), float(bias))['accuracy']
        best_accuracy = max(best_accuracy, float(accuracy))
    return round(best_accuracy, 6)


def _map_point(x_value: float, y_value: float, *, left: float, top: float, width: float, height: float) -> tuple[float, float]:
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    mapped_x = left + ((x_value - x_min) / (x_max - x_min) * width)
    mapped_y = top + height - ((y_value - y_min) / (y_max - y_min) * height)
    return mapped_x, mapped_y


def _boundary_endpoints(weights: tuple[float, float], bias: float) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5
    wx, wy = weights
    if abs(wy) < 1e-9:
        x_value = -bias / wx
        return (x_value, y_min), (x_value, y_max)
    y_left = (-(wx * x_min) - bias) / wy
    y_right = (-(wx * x_max) - bias) / wy
    return (x_min, y_left), (x_max, y_right)


def _panel_svg(
    *,
    title: str,
    subtitle: str,
    left: float,
    top: float,
    width: float,
    height: float,
    points: list[tuple[float, float]],
    labels: list[int],
    predictions: list[int],
    weights: tuple[float, float],
    bias: float,
) -> str:
    panel = [
        f'<text x="{left}" y="{top - 18}" font-size="20" font-family="Arial, sans-serif">{title}</text>',
        f'<text x="{left}" y="{top + 2}" font-size="13" font-family="Arial, sans-serif" fill="#555">{subtitle}</text>',
        f'<rect x="{left}" y="{top + 16}" width="{width}" height="{height}" fill="#f8f9fa" stroke="#ced4da" />',
    ]

    axis_top = top + 16
    axis_bottom = axis_top + height
    axis_left = left
    axis_right = left + width
    zero_x, zero_y = _map_point(0.0, 0.0, left=left, top=axis_top, width=width, height=height)
    panel.append(f'<line x1="{axis_left}" y1="{zero_y}" x2="{axis_right}" y2="{zero_y}" stroke="#adb5bd" stroke-width="1.5" />')
    panel.append(f'<line x1="{zero_x}" y1="{axis_top}" x2="{zero_x}" y2="{axis_bottom}" stroke="#adb5bd" stroke-width="1.5" />')

    boundary_start, boundary_end = _boundary_endpoints(weights, bias)
    line_start = _map_point(*boundary_start, left=left, top=axis_top, width=width, height=height)
    line_end = _map_point(*boundary_end, left=left, top=axis_top, width=width, height=height)
    panel.append(
        f'<line x1="{line_start[0]:.2f}" y1="{line_start[1]:.2f}" '
        f'x2="{line_end[0]:.2f}" y2="{line_end[1]:.2f}" stroke="#1c7ed6" stroke-width="3" />'
    )

    for point, label, prediction in zip(points, labels, predictions):
        mapped_x, mapped_y = _map_point(*point, left=left, top=axis_top, width=width, height=height)
        fill = '#d94841' if label == 1 else '#364fc7'
        stroke = '#111111' if prediction == label else '#f08c00'
        panel.append(
            f'<circle cx="{mapped_x:.2f}" cy="{mapped_y:.2f}" r="9" fill="{fill}" stroke="{stroke}" stroke-width="3" />'
        )
        panel.append(
            f'<text x="{mapped_x + 12:.2f}" y="{mapped_y - 10:.2f}" font-size="11" font-family="Arial, sans-serif">'
            f'gold={label}, pred={prediction}</text>'
        )

    return '\n'.join(panel)


def save_svg(linear_eval: dict[str, object], xor_eval: dict[str, object]) -> None:
    width, height = 980, 420
    panel_width, panel_height = 420, 280
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="40" y="36" font-size="26" font-family="Arial, sans-serif">Decision boundaries: perceptron vs XOR</text>
  <text x="40" y="62" font-size="14" font-family="Arial, sans-serif" fill="#555">파란 테두리는 정답, 주황 테두리는 오분류를 뜻한다.</text>
  {_panel_svg(
      title='선형 분리 가능 데이터',
      subtitle='x >= 0 경계 하나로 4개 점을 모두 맞춘다.',
      left=40,
      top=96,
      width=panel_width,
      height=panel_height,
      points=LINEAR_POINTS,
      labels=LINEAR_LABELS,
      predictions=linear_eval['predictions'],
      weights=(LINEAR_RULE['weights'][0], LINEAR_RULE['weights'][1]),
      bias=LINEAR_RULE['bias'],
  )}
  {_panel_svg(
      title='XOR 데이터',
      subtitle='single neuron이 줄 수 있는 최선도 한 점은 놓친다.',
      left=520,
      top=96,
      width=panel_width,
      height=panel_height,
      points=XOR_POINTS,
      labels=XOR_LABELS,
      predictions=xor_eval['predictions'],
      weights=(XOR_RULE['weights'][0], XOR_RULE['weights'][1]),
      bias=XOR_RULE['bias'],
  )}
  <rect x="40" y="392" width="14" height="14" fill="#364fc7" stroke="#111111" />
  <text x="62" y="404" font-size="13" font-family="Arial, sans-serif">label 0</text>
  <rect x="150" y="392" width="14" height="14" fill="#d94841" stroke="#111111" />
  <text x="172" y="404" font-size="13" font-family="Arial, sans-serif">label 1</text>
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    linear_eval = evaluate_dataset(LINEAR_POINTS, LINEAR_LABELS, LINEAR_RULE['weights'], LINEAR_RULE['bias'])
    xor_eval = evaluate_dataset(XOR_POINTS, XOR_LABELS, XOR_RULE['weights'], XOR_RULE['bias'])
    xor_best_accuracy = best_single_neuron_accuracy(XOR_POINTS, XOR_LABELS)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(linear_eval=linear_eval, xor_eval=xor_eval)

    metrics = {
        'decision_rule': 'predict=1 if w·x + b >= 0 else 0',
        'search_grid_values': list(GRID_VALUES),
        'linear_rule_weights': linear_eval['weights'],
        'linear_rule_bias': linear_eval['bias'],
        'linear_dataset_accuracy': linear_eval['accuracy'],
        'linear_dataset_scores': linear_eval['scores'],
        'linear_is_separable': linear_eval['accuracy'] == 1.0,
        'xor_example_weights': xor_eval['weights'],
        'xor_example_bias': xor_eval['bias'],
        'xor_example_accuracy': xor_eval['accuracy'],
        'xor_example_scores': xor_eval['scores'],
        'xor_best_accuracy': xor_best_accuracy,
        'xor_is_separable_with_single_neuron': xor_best_accuracy == 1.0,
        'xor_failure_reason': '직선 하나로는 XOR의 대각선 패턴을 동시에 나눌 수 없다.',
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
