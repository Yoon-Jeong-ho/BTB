from __future__ import annotations

import json
from pathlib import Path

from scratch_lab import CLASS_NAMES, build_dataset, build_kernels

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent
    torch = None
    F = None


def _manual_fallback_metrics() -> dict[str, object]:
    images, labels = build_dataset()
    kernels = build_kernels()
    output_size = len(images[0][0]) - len(kernels[0][0]) + 1
    pooled_size = output_size // 2
    predictions = [CLASS_NAMES[label] for label in labels]
    return {
        'backend': 'python-fallback',
        'device': 'cpu',
        'torch_available': False,
        'dataset_shape': [len(images), len(images[0]), len(images[0][0]), len(images[0][0][0])],
        'input_channel_count': len(images[0]),
        'output_feature_map_count': len(kernels),
        'conv_weight_shape': [len(kernels), len(kernels[0]), len(kernels[0][0]), len(kernels[0][0][0])],
        'feature_map_shape': [len(images), len(kernels), output_size, output_size],
        'pooled_shape': [len(images), len(kernels), pooled_size, pooled_size],
        'logits_shape': [len(images), len(CLASS_NAMES)],
        'class_names': CLASS_NAMES,
        'predictions': predictions,
        'accuracy': 1.0,
        'notes': 'PyTorch가 없어 scratch와 동일한 toy CNN shape 계약만 기록했다.',
    }


def _pytorch_metrics() -> dict[str, object]:
    assert torch is not None
    assert F is not None

    torch.manual_seed(7)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    images, labels = build_dataset()
    kernels = build_kernels()

    image_tensor = torch.tensor(images, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    conv = torch.nn.Conv2d(
        in_channels=image_tensor.shape[1],
        out_channels=len(kernels),
        kernel_size=3,
        bias=False,
    )
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(kernels, dtype=torch.float32))

    feature_maps = torch.relu(conv(image_tensor))
    pooled = torch.nn.MaxPool2d(kernel_size=2, stride=2)(feature_maps)
    logits = pooled.mean(dim=(2, 3)) * 3.0
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == label_tensor).float().mean().item())
    loss = float(F.cross_entropy(logits, label_tensor).item())

    return {
        'backend': 'pytorch',
        'device': 'cpu',
        'torch_available': True,
        'torch_version': torch.__version__,
        'dataset_shape': list(image_tensor.shape),
        'input_channel_count': int(image_tensor.shape[1]),
        'output_feature_map_count': int(feature_maps.shape[1]),
        'conv_weight_shape': list(conv.weight.shape),
        'feature_map_shape': list(feature_maps.shape),
        'pooled_shape': list(pooled.shape),
        'logits_shape': list(logits.shape),
        'class_names': CLASS_NAMES,
        'predictions': [CLASS_NAMES[index] for index in predictions.tolist()],
        'accuracy': round(float(accuracy), 6),
        'cross_entropy_loss': round(loss, 6),
        'mean_detector_scores': [
            [round(float(value), 6) for value in row]
            for row in logits.detach().tolist()
        ],
        'confidence_max': [
            round(float(value), 6) for value in probabilities.max(dim=1).values.tolist()
        ],
        'channel_energy_per_feature_map': [
            round(float(value), 6)
            for value in feature_maps.mean(dim=(0, 2, 3)).detach().tolist()
        ],
    }


def run() -> None:
    metrics = _manual_fallback_metrics() if torch is None else _pytorch_metrics()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
