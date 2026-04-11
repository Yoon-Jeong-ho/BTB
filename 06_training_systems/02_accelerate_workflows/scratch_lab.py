from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'accelerate_workflow.svg'


def save_svg() -> None:
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="620" height="180">
  <rect width="100%" height="100%" fill="#fff" />
  <text x="20" y="30" font-size="18">Accelerate workflow boundary</text>
  <rect x="40" y="70" width="140" height="50" fill="#dbeafe" stroke="#1d4ed8" />
  <text x="55" y="100" font-size="13">plain loop</text>
  <rect x="240" y="70" width="140" height="50" fill="#dcfce7" stroke="#15803d" />
  <text x="255" y="100" font-size="13">Accelerator</text>
  <rect x="440" y="70" width="140" height="50" fill="#fef3c7" stroke="#b45309" />
  <text x="455" y="100" font-size="13">backend</text>
  <line x1="180" y1="95" x2="240" y2="95" stroke="#222" marker-end="url(#a)" />
  <line x1="380" y1="95" x2="440" y2="95" stroke="#222" marker-end="url(#a)" />
</svg>'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    metrics = {
        'baseline_explicit_device_calls': 3,
        'baseline_manual_backward': True,
        'accelerate_replaced_calls': 3,
        'distributed_type': 'MULTI_GPU',
        'num_processes': 4,
        'device_placement': True,
        'mixed_precision': 'bf16',
        'still_user_responsible_for': ['effective_batch', 'checkpointing', 'metric_gathering'],
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
