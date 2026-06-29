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


def _polyline(
    points: list[tuple[float, float]],
    color: str,
    *,
    width: int = 3,
    dash: str = '',
) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attr} stroke-linejoin="round" stroke-linecap="round" points="{point_text}" />'
    )


def _format_loss(value: object) -> str:
    numeric = float(value)
    if abs(numeric) >= 100000:
        return f'{numeric:.2e}'
    return f'{numeric:.1f}'


def save_svg(series: dict[str, list[float]], runs: dict[str, dict[str, object]]) -> None:
    width, height = 1040, 660
    x_min, x_max = 0.0, float(STEPS - 1)
    colors = {
        'raw/no-reg': '#d94841',
        'normalized/no-reg': '#1c7ed6',
        'normalized+l2': '#2b8a3e',
    }
    labels = {
        'raw/no-reg': 'Raw features, no weight decay',
        'normalized/no-reg': 'Z-score normalized',
        'normalized+l2': 'Z-score normalized + L2',
    }

    def map_point(
        x_value: float,
        y_value: float,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        y_min: float,
        y_max: float,
    ) -> tuple[float, float]:
        x = left + ((x_value - x_min) / (x_max - x_min)) * (right - left)
        y = bottom - ((y_value - y_min) / (y_max - y_min)) * (bottom - top)
        return x, y

    def ticks(y_min: float, y_max: float, count: int) -> list[float]:
        if count <= 1:
            return [y_min]
        step = (y_max - y_min) / (count - 1)
        return [y_min + (step * index) for index in range(count)]

    def draw_axes(
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        y_min: float,
        y_max: float,
        y_tick_values: list[float],
        title: str,
        y_label: str,
    ) -> list[str]:
        parts = [
            f'<text x="{left}" y="{top - 18}" font-size="15" font-weight="700" '
            f'font-family="Arial, sans-serif" fill="#212529">{title}</text>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />',
            f'<text x="{(left + right) / 2 - 35:.2f}" y="{bottom + 44}" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#495057">SGD step</text>',
            f'<text transform="rotate(-90 {left - 54:.2f} {(top + bottom) / 2:.2f})" '
            f'x="{left - 54:.2f}" y="{(top + bottom) / 2:.2f}" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#495057">{y_label}</text>',
        ]
        for y_value in y_tick_values:
            _, y = map_point(0.0, y_value, left=left, right=right, top=top, bottom=bottom, y_min=y_min, y_max=y_max)
            tick_label = f'{y_value:.2f}' if (y_max - y_min) < 5 else f'{y_value:.0f}'
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1" />',
                    f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222" stroke-width="1" />',
                    f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="11" text-anchor="end" '
                    f'font-family="Arial, sans-serif" fill="#495057">{tick_label}</text>',
                ]
            )
        for step in range(STEPS):
            x, _ = map_point(float(step), y_min, left=left, right=right, top=top, bottom=bottom, y_min=y_min, y_max=y_max)
            parts.extend(
                [
                    f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}" stroke="#222" stroke-width="1" />',
                    f'<text x="{x:.2f}" y="{bottom + 20}" font-size="11" text-anchor="middle" '
                    f'font-family="Arial, sans-serif" fill="#495057">{step}</text>',
                ]
            )
        return parts

    def draw_series(
        keys: list[str],
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        y_min: float,
        y_max: float,
    ) -> list[str]:
        parts: list[str] = []
        for key in keys:
            points = [
                map_point(float(step), float(loss), left=left, right=right, top=top, bottom=bottom, y_min=y_min, y_max=y_max)
                for step, loss in enumerate(series[key])
            ]
            parts.append(_polyline(points, colors[key], width=3, dash='5 4' if key == 'normalized+l2' else ''))
            for x, y in points:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="#ffffff" stroke="{colors[key]}" stroke-width="2" />')
        return parts

    main_left, main_right = 82.0, 555.0
    zoom_left, zoom_right = 658.0, 982.0
    panel_top, panel_bottom = 105.0, 330.0
    all_values = [value for values in series.values() for value in values]
    main_y_min = 0.0
    main_y_max = math.ceil((max(all_values) + 1.0) / 5.0) * 5.0
    normalized_values = series['normalized/no-reg'] + series['normalized+l2']
    zoom_min = min(normalized_values)
    zoom_max = max(normalized_values)
    zoom_padding = max(0.04, (zoom_max - zoom_min) * 0.15)
    zoom_y_min = math.floor((zoom_min - zoom_padding) * 10.0) / 10.0
    zoom_y_max = math.ceil((zoom_max + zoom_padding) * 10.0) / 10.0

    main_parts = draw_axes(
        left=main_left,
        right=main_right,
        top=panel_top,
        bottom=panel_bottom,
        y_min=main_y_min,
        y_max=main_y_max,
        y_tick_values=[0, 5, 10, 15, 20, 25, main_y_max],
        title='A. Same scale: raw input explodes',
        y_label='log10(loss)',
    )
    main_parts.extend(
        draw_series(
            ['raw/no-reg', 'normalized/no-reg', 'normalized+l2'],
            left=main_left,
            right=main_right,
            top=panel_top,
            bottom=panel_bottom,
            y_min=main_y_min,
            y_max=main_y_max,
        )
    )
    raw_end_x, raw_end_y = map_point(
        float(STEPS - 1),
        series['raw/no-reg'][-1],
        left=main_left,
        right=main_right,
        top=panel_top,
        bottom=panel_bottom,
        y_min=main_y_min,
        y_max=main_y_max,
    )
    main_parts.extend(
        [
            f'<text x="{raw_end_x - 144:.2f}" y="{raw_end_y - 12:.2f}" font-size="12" font-weight="700" '
            f'font-family="Arial, sans-serif" fill="{colors["raw/no-reg"]}">raw final loss {_format_loss(runs["raw/no-reg"]["loss_history"][-1])}</text>',
            f'<rect x="{main_left + 12}" y="{panel_top + 12}" width="290" height="26" rx="8" fill="#ffffff" stroke="#e9ecef" />',
            f'<text x="{main_left + 24}" y="{panel_top + 30}" font-size="11.5" '
            f'font-family="Arial, sans-serif" fill="#495057">blue/green are visible in the zoom panel</text>',
        ]
    )

    zoom_parts = draw_axes(
        left=zoom_left,
        right=zoom_right,
        top=panel_top,
        bottom=panel_bottom,
        y_min=zoom_y_min,
        y_max=zoom_y_max,
        y_tick_values=ticks(zoom_y_min, zoom_y_max, 5),
        title='B. Zoom: normalized runs stay readable',
        y_label='log10(loss)',
    )
    zoom_parts.extend(
        draw_series(
            ['normalized/no-reg', 'normalized+l2'],
            left=zoom_left,
            right=zoom_right,
            top=panel_top,
            bottom=panel_bottom,
            y_min=zoom_y_min,
            y_max=zoom_y_max,
        )
    )
    for index, key in enumerate(['normalized/no-reg', 'normalized+l2']):
        label_y = 126 + (index * 24)
        zoom_parts.append(
            f'<rect x="{zoom_right - 170}" y="{label_y - 11}" width="13" height="13" rx="2" fill="{colors[key]}" />'
        )
        zoom_parts.append(
            f'<text x="{zoom_right - 150}" y="{label_y}" font-size="12" font-family="Arial, sans-serif" '
            f'fill="#212529">{labels[key]}</text>'
        )

    card_top = 435
    card_width = 292
    cards = [
        (
            82,
            colors['raw/no-reg'],
            'Raw features',
            [
                f'Initial |grad_w|: {runs["raw/no-reg"]["grad_history"][0]:.1f}',
                f'Final loss: {_format_loss(runs["raw/no-reg"]["loss_history"][-1])}',
                'Takeaway: update scale diverges',
            ],
        ),
        (
            374,
            colors['normalized/no-reg'],
            'Z-score normalized',
            [
                f'Initial |grad_w|: {runs["normalized/no-reg"]["grad_history"][0]:.2f}',
                f'Final loss: {_format_loss(runs["normalized/no-reg"]["loss_history"][-1])}',
                f'Final |w|: {runs["normalized/no-reg"]["final_weight_norm"]:.2f}',
            ],
        ),
        (
            666,
            colors['normalized+l2'],
            'Z-score + L2 decay',
            [
                f'Initial |grad_w|: {runs["normalized+l2"]["grad_history"][0]:.2f}',
                f'Final loss: {_format_loss(runs["normalized+l2"]["loss_history"][-1])}',
                f'L2 lowers |w| to {runs["normalized+l2"]["final_weight_norm"]:.2f}',
            ],
        ),
    ]
    card_parts: list[str] = []
    for x, color, title, rows in cards:
        card_parts.extend(
            [
                f'<rect x="{x}" y="{card_top}" width="{card_width}" height="124" rx="14" fill="#ffffff" '
                f'stroke="{color}" stroke-width="2" />',
                f'<text x="{x + 16}" y="{card_top + 28}" font-size="15" font-weight="700" '
                f'font-family="Arial, sans-serif" fill="{color}">{title}</text>',
            ]
        )
        for row_index, row in enumerate(rows):
            card_parts.append(
                f'<text x="{x + 16}" y="{card_top + 56 + (row_index * 22)}" font-size="13" '
                f'font-family="Arial, sans-serif" fill="#212529">{row}</text>'
            )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="46" y="34" font-size="22" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Training dynamics: normalization and regularization</text>
  <text x="46" y="58" font-size="13" font-family="Arial, sans-serif" fill="#495057">Same learning rate (eta = {LEARNING_RATE}); normalization stabilizes updates, while L2 mostly controls weight size.</text>
  <rect x="52" y="76" width="956" height="548" rx="20" fill="#f8fafc" stroke="#e9ecef" />
  {''.join(main_parts)}
  {''.join(zoom_parts)}
  <text x="82" y="410" font-size="14" font-weight="700" font-family="Arial, sans-serif" fill="#212529">How to read the result</text>
  <text x="250" y="410" font-size="12" font-family="Arial, sans-serif" fill="#495057">First compare raw vs normalized scale, then compare the regularizer's effect on final |w|.</text>
  {''.join(card_parts)}
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
        },
        {
            'raw/no-reg': raw_run,
            'normalized/no-reg': normalized_run,
            'normalized+l2': normalized_l2_run,
        },
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
