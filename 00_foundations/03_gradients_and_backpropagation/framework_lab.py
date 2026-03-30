from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


class TinyBackpropNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.layer1(inputs))
        return self.layer2(hidden)


def _round_list(values: torch.Tensor) -> list[float]:
    return [round(float(value), 6) for value in values.detach().cpu().view(-1)]


def run() -> None:
    torch.manual_seed(0)

    inputs = torch.tensor(
        [[1.0, -0.5], [0.3, 0.8], [-1.2, 0.4]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.4], [0.1], [-0.3]], dtype=torch.float32)

    model = TinyBackpropNet()
    with torch.no_grad():
        model.layer1.weight.copy_(torch.tensor([[0.5, -0.25], [-0.3, 0.8]], dtype=torch.float32))
        model.layer1.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        model.layer2.weight.copy_(torch.tensor([[1.2, -0.7]], dtype=torch.float32))
        model.layer2.bias.copy_(torch.tensor([0.05], dtype=torch.float32))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    optimizer.zero_grad()
    predictions_before = model(inputs)
    loss_before = criterion(predictions_before, targets)
    loss_before.backward()

    first_layer_weight_grad_norm = float(model.layer1.weight.grad.norm().item())
    first_layer_bias_grad_norm = float(model.layer1.bias.grad.norm().item())
    second_layer_weight_grad_norm = float(model.layer2.weight.grad.norm().item())
    second_layer_bias_grad_norm = float(model.layer2.bias.grad.norm().item())

    total_grad_norm = (
        (first_layer_weight_grad_norm**2)
        + (first_layer_bias_grad_norm**2)
        + (second_layer_weight_grad_norm**2)
        + (second_layer_bias_grad_norm**2)
    ) ** 0.5

    optimizer.step()

    with torch.no_grad():
        predictions_after = model(inputs)
        loss_after = criterion(predictions_after, targets)

    metrics = {
        'input_shape': list(inputs.shape),
        'target_shape': list(targets.shape),
        'loss_before_step': round(float(loss_before.item()), 6),
        'loss_after_step': round(float(loss_after.item()), 6),
        'first_prediction_before': round(float(predictions_before[0, 0].item()), 6),
        'first_prediction_after': round(float(predictions_after[0, 0].item()), 6),
        'predictions_before': _round_list(predictions_before),
        'predictions_after': _round_list(predictions_after),
        'first_layer_weight_grad_norm': round(first_layer_weight_grad_norm, 6),
        'first_layer_bias_grad_norm': round(first_layer_bias_grad_norm, 6),
        'second_layer_weight_grad_norm': round(second_layer_weight_grad_norm, 6),
        'second_layer_bias_grad_norm': round(second_layer_bias_grad_norm, 6),
        'total_grad_norm': round(total_grad_norm, 6),
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
