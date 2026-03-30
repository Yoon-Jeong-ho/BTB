from __future__ import annotations

import json
import time
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement()


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(param.nelement() * param.element_size() for param in model.parameters())


def gradient_bytes(model: torch.nn.Module) -> int:
    total = 0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.nelement() * param.grad.element_size()
    return total


def sync_if_needed(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()


def warmup(model: torch.nn.Module, batch: torch.Tensor, device: str) -> None:
    with torch.no_grad():
        _ = model(batch)
    sync_if_needed(device)

    model.zero_grad(set_to_none=True)
    warmup_loss = model(batch).pow(2).mean()
    warmup_loss.backward()
    sync_if_needed(device)
    model.zero_grad(set_to_none=True)


def run() -> None:
    torch.manual_seed(7)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
    ).to(device)
    batch = torch.randn(16, 1024, device=device)

    warmup(model, batch, device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync_if_needed(device)
    inference_start = time.perf_counter()
    with torch.no_grad():
        inference_output = model(batch)
    sync_if_needed(device)
    inference_runtime_ms = round((time.perf_counter() - inference_start) * 1000, 4)
    inference_allocated = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0
    inference_reserved = int(torch.cuda.max_memory_reserved()) if device == 'cuda' else 0

    model.zero_grad(set_to_none=True)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync_if_needed(device)
    training_start = time.perf_counter()
    training_output = model(batch)
    loss = training_output.pow(2).mean()
    loss.backward()
    sync_if_needed(device)
    training_runtime_ms = round((time.perf_counter() - training_start) * 1000, 4)
    training_allocated = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0
    training_reserved = int(torch.cuda.max_memory_reserved()) if device == 'cuda' else 0

    metrics = {
        'device': device,
        'dtype': str(batch.dtype),
        'batch_shape': list(batch.shape),
        'parameter_count': sum(param.numel() for param in model.parameters()),
        'parameter_bytes': parameter_bytes(model),
        'inference_runtime_ms': inference_runtime_ms,
        'training_runtime_ms': training_runtime_ms,
        'inference_output_bytes': tensor_nbytes(inference_output),
        'training_output_bytes': tensor_nbytes(training_output),
        'training_grad_bytes': gradient_bytes(model),
        'loss': round(float(loss.detach().cpu()), 6),
        'max_memory_allocated': training_allocated,
        'max_memory_reserved': training_reserved,
        'inference_max_memory_allocated': inference_allocated,
        'inference_max_memory_reserved': inference_reserved,
        'training_max_memory_allocated': training_allocated,
        'training_max_memory_reserved': training_reserved,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
