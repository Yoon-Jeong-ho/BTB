from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def run() -> None:
    prepared = ['model', 'optimizer', 'dataloader', 'scheduler']
    metrics = {
        'backend': 'accelerate-simulated',
        'prepared_object_count': len(prepared),
        'prepared_objects': prepared,
        'manual_rank_logic_removed': 'partially',
        'sync_gradients': True,
        'optimizer_step_was_skipped': False,
        'dataloader_behavior': 'sharded_and_device_placed',
        'remaining_complexity_count': 3,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
