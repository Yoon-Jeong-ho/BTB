from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'rank_gradients.svg'

RANKS = [
    {'rank': 0, 'local_rank': 0, 'node': 0, 'gradient': 1.00},
    {'rank': 1, 'local_rank': 1, 'node': 0, 'gradient': 1.20},
    {'rank': 2, 'local_rank': 0, 'node': 1, 'gradient': 0.80},
    {'rank': 3, 'local_rank': 1, 'node': 1, 'gradient': 1.10},
]


def save_svg() -> None:
    width, height = 560, 260
    left, bottom = 70, 210
    bar_w = 70
    max_grad = 1.3
    bars = []
    for idx, item in enumerate(RANKS):
        h = item['gradient'] / max_grad * 150
        x = left + idx * 110
        y = bottom - h
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="#2563eb" />')
        bars.append(f'<text x="{x}" y="230" font-size="12">rank {item["rank"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fff" />
  <text x="20" y="30" font-size="18">Rank-local gradients before all-reduce</text>
  <line x1="50" y1="{bottom}" x2="520" y2="{bottom}" stroke="#222" />
  {''.join(bars)}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    gradients = [r['gradient'] for r in RANKS]
    avg = sum(gradients) / len(gradients)
    metrics = {
        'world_size': len(RANKS),
        'local_world_size': len({r['local_rank'] for r in RANKS}),
        'node_count': len({r['node'] for r in RANKS}),
        'rank_to_local_rank': {str(r['rank']): r['local_rank'] for r in RANKS},
        'local_gradients': gradients,
        'averaged_gradient': round(avg, 6),
        'max_gradient_deviation': round(max(abs(g - avg) for g in gradients), 6),
        'all_reduce_operation': 'mean',
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
