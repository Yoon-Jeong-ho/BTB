from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'


def run() -> None:
    a = np.arange(6).reshape(2, 3)
    b = np.arange(12).reshape(3, 4)
    c = a @ b

    broadcast_source = np.array([[10.0], [20.0]])
    broadcast_result = a + broadcast_source

    mismatch_error = ''
    try:
        _ = a + np.ones((4,))
    except ValueError as exc:
        mismatch_error = str(exc)

    metrics = {
        'a_shape': list(a.shape),
        'b_shape': list(b.shape),
        'matmul_shape': list(c.shape),
        'broadcast_source_shape': list(broadcast_source.shape),
        'broadcast_result_shape': list(broadcast_result.shape),
        'mismatch_error': mismatch_error,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
