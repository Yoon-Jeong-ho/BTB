from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def run() -> None:
    gradients = [1.0, 1.2, 0.8, 1.1]
    averaged = sum(gradients) / len(gradients)
    parameter_before = 2.0
    learning_rate = 0.1
    parameter_after = parameter_before - learning_rate * averaged
    metrics = {
        'backend': 'cpu-simulated-ddp',
        'world_size': len(gradients),
        'gradient_vector_shape': [len(gradients)],
        'averaged_gradient': round(averaged, 6),
        'parameter_before': round(parameter_before, 6),
        'parameter_after': round(parameter_after, 6),
        'all_ranks_share_update': True,
        'rank_losses': [round(0.5 * g * g, 6) for g in gradients],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
