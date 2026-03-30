from __future__ import annotations

import json
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'


def run() -> None:
    torch.manual_seed(7)

    batch = torch.randn(4, 8)
    layer = torch.nn.Linear(8, 3)
    logits = layer(batch)
    probs = torch.softmax(logits, dim=-1)

    metrics = {
        'batch_shape': list(batch.shape),
        'weight_shape': list(layer.weight.shape),
        'bias_shape': list(layer.bias.shape),
        'logits_shape': list(logits.shape),
        'probs_shape': list(probs.shape),
        'row_probability_sums': [round(float(value), 6) for value in probs.sum(dim=-1).detach()],
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
