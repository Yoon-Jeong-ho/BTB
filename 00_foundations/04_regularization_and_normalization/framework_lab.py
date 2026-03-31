from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _rounded_list(values: torch.Tensor) -> list[float]:
    return [_round_float(value) for value in values.detach().cpu().view(-1)]


def run_weight_decay_step(weight_decay: float) -> dict[str, float | list[float]]:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.8, -0.4]], dtype=torch.float32))
        model.bias.copy_(torch.tensor([0.2], dtype=torch.float32))

    features = torch.tensor(
        [[1.0, -1.0], [0.5, 0.2], [-0.3, 0.8]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.7], [0.1], [-0.4]], dtype=torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=weight_decay)
    optimizer.zero_grad()
    predictions = model(features)
    loss = F.mse_loss(predictions, targets)
    loss.backward()

    weight_before = model.weight.detach().clone()
    bias_before = model.bias.detach().clone()
    grad_norm = model.weight.grad.detach().norm()

    optimizer.step()

    return {
        'loss': _round_float(loss.item()),
        'weight_before': _rounded_list(weight_before),
        'bias_before': _rounded_list(bias_before),
        'grad_norm': _round_float(grad_norm.item()),
        'weight_norm_before_step': _round_float(weight_before.norm().item()),
        'weight_norm_after_step': _round_float(model.weight.detach().norm().item()),
        'weight_after': _rounded_list(model.weight.detach()),
    }


def run() -> None:
    torch.manual_seed(123)

    inputs = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
        dtype=torch.float32,
    )

    layer_norm = torch.nn.LayerNorm(4, elementwise_affine=False, eps=0.0)
    normalized = layer_norm(inputs)

    dropout = torch.nn.Dropout(p=0.5)
    dropout.train()
    torch.manual_seed(123)
    dropout_train = dropout(inputs)
    dropout.eval()
    dropout_eval = dropout(inputs)

    no_weight_decay = run_weight_decay_step(weight_decay=0.0)
    with_weight_decay = run_weight_decay_step(weight_decay=0.2)

    metrics = {
        'input_shape': list(inputs.shape),
        'layernorm_row_means': _rounded_list(normalized.mean(dim=-1)),
        'layernorm_row_vars': _rounded_list(normalized.var(dim=-1, unbiased=False)),
        'dropout_train_zero_fraction': _round_float(float((dropout_train == 0).float().mean().item())),
        'dropout_eval_matches_input': bool(torch.equal(dropout_eval, inputs)),
        'dropout_train_output': _rounded_list(dropout_train),
        'dropout_eval_output': _rounded_list(dropout_eval),
        'no_weight_decay_loss': no_weight_decay['loss'],
        'no_weight_decay_grad_norm': no_weight_decay['grad_norm'],
        'no_weight_decay_weight_norm_before_step': no_weight_decay['weight_norm_before_step'],
        'no_weight_decay_weight_norm_after_step': no_weight_decay['weight_norm_after_step'],
        'weight_decay_loss': with_weight_decay['loss'],
        'weight_decay_grad_norm': with_weight_decay['grad_norm'],
        'weight_decay_weight_norm_before_step': with_weight_decay['weight_norm_before_step'],
        'weight_decay_weight_norm_after_step': with_weight_decay['weight_norm_after_step'],
        'weight_decay_delta': _round_float(
            no_weight_decay['weight_norm_after_step'] - with_weight_decay['weight_norm_after_step']
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
