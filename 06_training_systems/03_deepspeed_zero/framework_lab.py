from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def run() -> None:
    stage_memory = {'zero1': 48.0, 'zero2': 36.0, 'zero3': 24.0}
    metrics = {
        'backend': 'zero-simulated',
        'stage_memory_mb': stage_memory,
        'best_memory_stage': min(stage_memory, key=stage_memory.get),
        'communication_penalty_rank': [1, 2, 3],
        'checkpoint_complexity': {'zero1': 'low', 'zero2': 'medium', 'zero3': 'high'},
        'optimizer_state_partitioned': True,
        'parameters_partitioned_at_stage_3': True,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
