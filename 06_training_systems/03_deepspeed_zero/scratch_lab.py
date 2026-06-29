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
    width, height = 820, 390
    left, right = 82, 770
    top, bottom = 82, 285
    bars = []
    grid = []
    max_v = 100.0
    labels = {
        'dp_baseline': ('DP baseline', 'all states replicated'),
        'zero_stage_1': ('ZeRO stage 1', 'optimizer state sharded'),
        'zero_stage_2': ('ZeRO stage 2', 'optimizer + grads sharded'),
        'zero_stage_3': ('ZeRO stage 3', 'params + grads + optimizer sharded'),
    }
    for tick in [0, 25, 50, 75, 100]:
        y = bottom - (tick / max_v * (bottom - top))
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#edf2f7" />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#495057">{tick}</text>')
    for i, (name, value) in enumerate(values.items()):
        h = value / max_v * (bottom - top)
        x = left + 35 + i * 166
        y = bottom - h
        title, subtitle = labels[name]
        color = '#7c3aed' if name != 'zero_stage_3' else '#2b8a3e'
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="92" height="{h:.2f}" rx="8" fill="{color}" />')
        bars.append(f'<text x="{x + 46}" y="{y - 10:.2f}" text-anchor="middle" font-size="13" font-weight="700" font-family="Arial, sans-serif" fill="{color}">{value:.0f} MB</text>')
        bars.append(f'<text x="{x + 46}" y="{bottom + 23}" text-anchor="middle" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#212529">{title}</text>')
        bars.append(f'<text x="{x + 46}" y="{bottom + 42}" text-anchor="middle" font-size="10.5" font-family="Arial, sans-serif" fill="#495057">{subtitle}</text>')
    stage_3_saving = 1.0 - values['zero_stage_3'] / values['dp_baseline']
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="38" y="34" font-size="21" font-weight="700" font-family="Arial, sans-serif" fill="#111827">ZeRO stage memory per rank</text>
  <text x="38" y="57" font-size="13" font-family="Arial, sans-serif" fill="#495057">Each stage shards more model state across {WORLD_SIZE} ranks, so one rank stores less memory.</text>
  <rect x="38" y="68" width="748" height="285" rx="18" fill="#f8fafc" stroke="#e9ecef"/>
  {''.join(grid)}
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <text transform="rotate(-90 {left - 54} {(top + bottom) / 2:.2f})" x="{left - 54}" y="{(top + bottom) / 2:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#495057">Memory per rank (MB)</text>
  {''.join(bars)}
  <rect x="565" y="93" width="188" height="66" rx="12" fill="#ffffff" stroke="#c3e6cb"/>
  <text x="581" y="121" font-size="13" font-weight="700" font-family="Arial, sans-serif" fill="#2b8a3e">Stage 3 = {stage_3_saving:.0%} less memory</text>
  <text x="581" y="145" font-size="11" font-family="Arial, sans-serif" fill="#495057">shards params + grads + optimizer</text>
</svg>'''
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
