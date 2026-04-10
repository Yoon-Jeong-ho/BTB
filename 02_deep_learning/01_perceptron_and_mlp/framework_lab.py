from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'

LINEAR_INPUTS = [[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]]
LINEAR_TARGETS = [[0.0], [0.0], [1.0], [1.0]]
XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_TARGETS = [[0.0], [1.0], [1.0], [0.0]]


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _manual_step(score: float) -> float:
    return 1.0 if score >= 0.0 else 0.0


def _python_fallback_metrics() -> dict[str, object]:
    linear_predictions = [_manual_step(point[0]) for point in LINEAR_INPUTS]
    single_xor_predictions = [_manual_step(point[0] + point[1] - 0.5) for point in XOR_INPUTS]

    def tiny_mlp_predict(point: list[float]) -> float:
        x1, x2 = point
        hidden_or = _manual_step(x1 + x2 - 0.5)
        hidden_and = _manual_step(x1 + x2 - 1.5)
        return _manual_step(hidden_or - (2.0 * hidden_and) - 0.5)

    mlp_predictions = [tiny_mlp_predict(point) for point in XOR_INPUTS]
    linear_accuracy = sum(int(pred == gold[0]) for pred, gold in zip(linear_predictions, LINEAR_TARGETS)) / len(LINEAR_TARGETS)
    single_xor_accuracy = sum(int(pred == gold[0]) for pred, gold in zip(single_xor_predictions, XOR_TARGETS)) / len(XOR_TARGETS)
    mlp_xor_accuracy = sum(int(pred == gold[0]) for pred, gold in zip(mlp_predictions, XOR_TARGETS)) / len(XOR_TARGETS)

    return {
        'backend': 'python-fallback',
        'device': 'cpu',
        'seed': None,
        'torch_available': False,
        'single_neuron_linear_accuracy': round(linear_accuracy, 6),
        'single_neuron_linear_loss': None,
        'single_neuron_xor_accuracy': round(single_xor_accuracy, 6),
        'single_neuron_xor_loss': None,
        'single_neuron_xor_probabilities': _round_list(single_xor_predictions),
        'single_neuron_parameter_count': 3,
        'tiny_mlp_xor_accuracy': round(mlp_xor_accuracy, 6),
        'tiny_mlp_xor_loss': None,
        'tiny_mlp_xor_probabilities': _round_list(mlp_predictions),
        'tiny_mlp_parameter_count': 9,
        'xor_accuracy_gain': round(mlp_xor_accuracy - single_xor_accuracy, 6),
        'notes': 'PyTorch가 없어 threshold 기반 fallback 관측만 남겼다.',
    }


try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None


def _train_binary_classifier(
    model: 'nn.Module',
    inputs: 'torch.Tensor',
    targets: 'torch.Tensor',
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_start = None
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        if loss_start is None:
            loss_start = float(loss.detach().item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = float((predictions == targets).float().mean().item())
        final_loss = float(criterion(logits, targets).item())

    return {
        'accuracy': round(accuracy, 6),
        'loss_start': round(float(loss_start), 6),
        'loss_final': round(final_loss, 6),
        'probabilities': _round_list(probabilities.detach().cpu().view(-1).tolist()),
        'predictions': [int(value) for value in predictions.detach().cpu().view(-1).tolist()],
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
    }


def _pytorch_metrics() -> dict[str, object]:
    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    linear_inputs = torch.tensor(LINEAR_INPUTS, dtype=torch.float32)
    linear_targets = torch.tensor(LINEAR_TARGETS, dtype=torch.float32)
    xor_inputs = torch.tensor(XOR_INPUTS, dtype=torch.float32)
    xor_targets = torch.tensor(XOR_TARGETS, dtype=torch.float32)

    torch.manual_seed(0)
    single_linear = nn.Linear(2, 1)
    linear_metrics = _train_binary_classifier(
        single_linear,
        linear_inputs,
        linear_targets,
        steps=400,
        learning_rate=0.2,
    )

    torch.manual_seed(0)
    single_xor = nn.Linear(2, 1)
    single_xor_metrics = _train_binary_classifier(
        single_xor,
        xor_inputs,
        xor_targets,
        steps=2000,
        learning_rate=0.2,
    )

    torch.manual_seed(0)
    tiny_mlp = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    mlp_metrics = _train_binary_classifier(
        tiny_mlp,
        xor_inputs,
        xor_targets,
        steps=4000,
        learning_rate=0.2,
    )

    return {
        'backend': 'pytorch',
        'device': 'cpu',
        'seed': 0,
        'torch_available': True,
        'torch_version': torch.__version__,
        'single_neuron_linear_accuracy': linear_metrics['accuracy'],
        'single_neuron_linear_loss': linear_metrics['loss_final'],
        'single_neuron_linear_probabilities': linear_metrics['probabilities'],
        'single_neuron_xor_accuracy': single_xor_metrics['accuracy'],
        'single_neuron_xor_loss': single_xor_metrics['loss_final'],
        'single_neuron_xor_probabilities': single_xor_metrics['probabilities'],
        'single_neuron_parameter_count': linear_metrics['parameter_count'],
        'tiny_mlp_xor_accuracy': mlp_metrics['accuracy'],
        'tiny_mlp_xor_loss': mlp_metrics['loss_final'],
        'tiny_mlp_xor_probabilities': mlp_metrics['probabilities'],
        'tiny_mlp_parameter_count': mlp_metrics['parameter_count'],
        'xor_accuracy_gain': round(mlp_metrics['accuracy'] - single_xor_metrics['accuracy'], 6),
        'single_neuron_xor_loss_start': single_xor_metrics['loss_start'],
        'tiny_mlp_xor_loss_start': mlp_metrics['loss_start'],
    }


def run() -> None:
    metrics = _python_fallback_metrics() if torch is None else _pytorch_metrics()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
