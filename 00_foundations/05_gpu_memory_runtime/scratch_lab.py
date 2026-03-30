from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
DTYPE_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2}


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total * DTYPE_BYTES[dtype]


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 4)


def run() -> None:
    batch_shape = (32, 512, 768)
    hidden_shape = (32, 512, 3072)

    batch_fp32_bytes = tensor_bytes(batch_shape, 'fp32')
    batch_fp16_bytes = tensor_bytes(batch_shape, 'fp16')
    hidden_fp32_bytes = tensor_bytes(hidden_shape, 'fp32')
    hidden_fp16_bytes = tensor_bytes(hidden_shape, 'fp16')

    metrics = {
        'batch_shape': list(batch_shape),
        'hidden_shape': list(hidden_shape),
        'batch_fp32_bytes': batch_fp32_bytes,
        'batch_fp16_bytes': batch_fp16_bytes,
        'batch_fp32_mib': mib(batch_fp32_bytes),
        'batch_fp16_mib': mib(batch_fp16_bytes),
        'hidden_fp32_bytes': hidden_fp32_bytes,
        'hidden_fp16_bytes': hidden_fp16_bytes,
        'dtype_savings_bytes': batch_fp32_bytes - batch_fp16_bytes,
        'dtype_savings_ratio': round(batch_fp32_bytes / batch_fp16_bytes, 2),
        'estimated_training_extra_bytes': hidden_fp32_bytes,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
