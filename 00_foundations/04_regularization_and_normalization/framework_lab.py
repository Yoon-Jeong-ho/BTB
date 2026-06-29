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
    data_weight_grad = model.weight.grad.detach().clone()
    decay_term = weight_decay * weight_before
    effective_weight_grad = data_weight_grad + decay_term
    grad_norm = data_weight_grad.norm()
    regularized_objective = loss + (0.5 * weight_decay * torch.sum(weight_before**2))

    optimizer.step()
    post_step_loss = F.mse_loss(model(features), targets)
    weight_after = model.weight.detach().clone()

    return {
        'weight_decay': _round_float(weight_decay),
        'loss': _round_float(loss.item()),
        'data_loss_before_step': _round_float(loss.item()),
        'regularized_objective_before_step': _round_float(regularized_objective.item()),
        'post_step_data_loss': _round_float(post_step_loss.item()),
        'weight_before': _rounded_list(weight_before),
        'bias_before': _rounded_list(bias_before),
        'data_weight_grad_before_decay': _rounded_list(data_weight_grad),
        'decay_term_added_to_weight_grad': _rounded_list(decay_term),
        'effective_weight_grad': _rounded_list(effective_weight_grad),
        'grad_norm': _round_float(grad_norm.item()),
        'data_grad_norm_before_decay': _round_float(grad_norm.item()),
        'decay_term_norm': _round_float(decay_term.norm().item()),
        'effective_weight_grad_norm': _round_float(effective_weight_grad.norm().item()),
        'weight_norm_before_step': _round_float(weight_before.norm().item()),
        'weight_norm_after_step': _round_float(weight_after.norm().item()),
        'weight_norm_shrinkage': _round_float(weight_before.norm().item() - weight_after.norm().item()),
        'weight_after': _rounded_list(weight_after),
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
        'weight_decay_strength': with_weight_decay['weight_decay'],
        'no_weight_decay_loss': no_weight_decay['loss'],
        'no_weight_decay_data_loss_before_step': no_weight_decay['data_loss_before_step'],
        'no_weight_decay_regularized_objective_before_step': no_weight_decay['regularized_objective_before_step'],
        'no_weight_decay_post_step_data_loss': no_weight_decay['post_step_data_loss'],
        'no_weight_decay_grad_norm': no_weight_decay['grad_norm'],
        'no_weight_decay_decay_term_norm': no_weight_decay['decay_term_norm'],
        'no_weight_decay_effective_weight_grad_norm': no_weight_decay['effective_weight_grad_norm'],
        'no_weight_decay_weight_norm_before_step': no_weight_decay['weight_norm_before_step'],
        'no_weight_decay_weight_norm_after_step': no_weight_decay['weight_norm_after_step'],
        'no_weight_decay_weight_norm_shrinkage': no_weight_decay['weight_norm_shrinkage'],
        'no_weight_decay_weight_after': no_weight_decay['weight_after'],
        'weight_decay_loss': with_weight_decay['loss'],
        'weight_decay_data_loss_before_step': with_weight_decay['data_loss_before_step'],
        'weight_decay_regularized_objective_before_step': with_weight_decay['regularized_objective_before_step'],
        'weight_decay_post_step_data_loss': with_weight_decay['post_step_data_loss'],
        'weight_decay_grad_norm': with_weight_decay['grad_norm'],
        'weight_decay_decay_term_norm': with_weight_decay['decay_term_norm'],
        'weight_decay_effective_weight_grad_norm': with_weight_decay['effective_weight_grad_norm'],
        'weight_decay_decay_term_added_to_weight_grad': with_weight_decay['decay_term_added_to_weight_grad'],
        'weight_decay_weight_norm_before_step': with_weight_decay['weight_norm_before_step'],
        'weight_decay_weight_norm_after_step': with_weight_decay['weight_norm_after_step'],
        'weight_decay_weight_norm_shrinkage': with_weight_decay['weight_norm_shrinkage'],
        'weight_decay_weight_after': with_weight_decay['weight_after'],
        'weight_decay_delta': _round_float(
            no_weight_decay['weight_norm_after_step'] - with_weight_decay['weight_norm_after_step']
        ),
        'post_step_data_loss_delta': _round_float(
            no_weight_decay['post_step_data_loss'] - with_weight_decay['post_step_data_loss']
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
