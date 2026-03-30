from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def run() -> None:
    torch.manual_seed(7)

    activation_inputs = torch.tensor(
        [[-2.0, -0.5, 0.0, 0.5, 2.0], [1.5, -1.0, 0.25, -0.25, 0.75]],
        dtype=torch.float32,
    )
    relu_values = torch.relu(activation_inputs)
    sigmoid_values = torch.sigmoid(activation_inputs)
    tanh_values = torch.tanh(activation_inputs)

    class_logits = torch.tensor(
        [[2.2, 0.3, -1.4], [0.1, 1.7, -0.5]],
        dtype=torch.float32,
    )
    class_targets = torch.tensor([0, 1], dtype=torch.long)
    class_probabilities = torch.softmax(class_logits, dim=-1)
    cross_entropy_loss = F.cross_entropy(class_logits, class_targets)

    binary_logits = torch.tensor([1.25, -0.75, 0.2, -1.5], dtype=torch.float32)
    binary_targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    binary_probabilities = torch.sigmoid(binary_logits)
    binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)

    metrics = {
        'activation_input_shape': list(activation_inputs.shape),
        'relu_zero_fraction': round(float((relu_values == 0).float().mean()), 6),
        'relu_first_row': [round(float(value), 6) for value in relu_values[0]],
        'sigmoid_first_row': [round(float(value), 6) for value in sigmoid_values[0]],
        'tanh_first_row': [round(float(value), 6) for value in tanh_values[0]],
        'class_logits_shape': list(class_logits.shape),
        'row_probability_sums': [round(float(value), 6) for value in class_probabilities.sum(dim=-1)],
        'target_class_probabilities': [
            round(float(class_probabilities[index, target]), 6)
            for index, target in enumerate(class_targets.tolist())
        ],
        'cross_entropy_loss': round(float(cross_entropy_loss), 6),
        'binary_probabilities': [round(float(value), 6) for value in binary_probabilities],
        'binary_cross_entropy_loss': round(float(binary_cross_entropy_loss), 6),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
