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


def save_svg(averaged_gradient: float) -> None:
    width, height = 760, 360
    left, right = 78, 700
    top, bottom = 74, 275
    bar_w = 78
    max_grad = 1.35
    bars = []
    grid = []
    for tick in [0.0, 0.5, 1.0, averaged_gradient]:
        y = bottom - (tick / max_grad * (bottom - top))
        color = '#adb5bd' if tick == averaged_gradient else '#edf2f7'
        dash = ' stroke-dasharray="5 4"' if tick == averaged_gradient else ''
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{color}"{dash} />')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#495057">{tick:.3g}</text>')
    for idx, item in enumerate(RANKS):
        h = item['gradient'] / max_grad * (bottom - top)
        x = left + 26 + idx * 142
        y = bottom - h
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" rx="7" fill="#2563eb" />')
        bars.append(f'<text x="{x + (bar_w / 2):.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#1e3a8a">{item["gradient"]:.2f}</text>')
        bars.append(f'<text x="{x + (bar_w / 2):.2f}" y="{bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#212529">rank {item["rank"]}</text>')
        bars.append(f'<text x="{x + (bar_w / 2):.2f}" y="{bottom + 40}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#495057">node {item["node"]} / local {item["local_rank"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fff" />
  <text x="34" y="32" font-size="21" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Rank-local gradients before all-reduce</text>
  <text x="34" y="55" font-size="13" font-family="Arial, sans-serif" fill="#495057">Local gradient before sync differs by rank; all-reduce replaces them with the same mean.</text>
  <rect x="34" y="68" width="692" height="260" rx="18" fill="#f8fafc" stroke="#e9ecef" />
  {''.join(grid)}
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <text transform="rotate(-90 {left - 52} {(top + bottom) / 2:.2f})" x="{left - 52}" y="{(top + bottom) / 2:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#495057">local gradient value</text>
  {''.join(bars)}
  <rect x="{left + 250}" y="{bottom - (averaged_gradient / max_grad * (bottom - top)) - 27:.2f}" width="166" height="20" rx="6" fill="#ffffff" stroke="#e9ecef" />
  <text x="{left + 260}" y="{bottom - (averaged_gradient / max_grad * (bottom - top)) - 12:.2f}" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#495057">All-reduce mean = {averaged_gradient:.3f}</text>
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
    save_svg(avg)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
