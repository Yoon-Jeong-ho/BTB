from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'zero_memory_stages.svg'
WORLD_SIZE = 4
COMPONENTS = {'parameters_mb': 24.0, 'gradients_mb': 24.0, 'optimizer_state_mb': 48.0}


def stage_memory(stage: int) -> float:
    params = COMPONENTS['parameters_mb'] / (WORLD_SIZE if stage >= 3 else 1)
    grads = COMPONENTS['gradients_mb'] / (WORLD_SIZE if stage >= 2 else 1)
    opt = COMPONENTS['optimizer_state_mb'] / (WORLD_SIZE if stage >= 1 else 1)
    return params + grads + opt


def save_svg(values: dict[str, float]) -> None:
    width, height = 620, 260
    bars = []
    max_v = max(values.values())
    for i, (name, value) in enumerate(values.items()):
        h = value / max_v * 150
        x = 60 + i * 135
        y = 210 - h
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="80" height="{h:.2f}" fill="#7c3aed" />')
        bars.append(f'<text x="{x}" y="232" font-size="12">{name}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#fff"/><text x="20" y="30" font-size="18">ZeRO stage memory per rank</text>{"".join(bars)}</svg>'
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    values = {
        'dp_baseline': sum(COMPONENTS.values()),
        'zero_stage_1': stage_memory(1),
        'zero_stage_2': stage_memory(2),
        'zero_stage_3': stage_memory(3),
    }
    metrics = {
        'world_size': WORLD_SIZE,
        'components_mb': COMPONENTS,
        'dp_baseline_mb': round(values['dp_baseline'], 6),
        'zero_stage_1_mb': round(values['zero_stage_1'], 6),
        'zero_stage_2_mb': round(values['zero_stage_2'], 6),
        'zero_stage_3_mb': round(values['zero_stage_3'], 6),
        'stage_3_memory_saving_ratio': round(1.0 - values['zero_stage_3'] / values['dp_baseline'], 6),
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(values)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
