from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
SEED = 17
INPUT_DIM = 8
BOTTLENECK_LATENT_DIM = 3
NARROW_LATENT_DIM = 1

MEAN_VECTOR = [0.45, 0.55, 0.5, 0.6, 0.35, 0.45, 0.4, 0.5]
PRIMARY_BASIS_RAW = [
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
]
NOISE_BASIS_RAW = [
    [1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0],
]
LATENT_CODES = [
    [1.0, 0.2, -0.4],
    [0.8, -0.3, 0.6],
    [0.6, 0.9, 0.2],
    [1.2, 0.7, -0.1],
    [0.4, -0.6, 0.9],
    [1.1, 0.4, 0.5],
    [-0.2, 1.0, -0.7],
    [0.3, -0.9, -0.2],
]
NOISE_CODES = [
    [0.18, -0.10],
    [-0.12, 0.14],
    [0.08, -0.16],
    [-0.15, -0.12],
    [0.16, 0.11],
    [-0.09, 0.17],
    [0.13, -0.08],
    [-0.11, -0.15],
]


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=1, keepdim=True)


def _make_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(MEAN_VECTOR, dtype=torch.float32)
    basis = _normalize_rows(torch.tensor(PRIMARY_BASIS_RAW, dtype=torch.float32))
    noise_basis = _normalize_rows(torch.tensor(NOISE_BASIS_RAW, dtype=torch.float32))
    latent = torch.tensor(LATENT_CODES, dtype=torch.float32)
    noise_codes = torch.tensor(NOISE_CODES, dtype=torch.float32)
    clean = mean.unsqueeze(0) + latent @ basis
    noisy = clean + noise_codes @ noise_basis
    return clean, noisy


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(inputs)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


def _train_autoencoder(
    train_inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    latent_dim: int,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = TinyAutoencoder(INPUT_DIM, latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_start = None

    for _ in range(steps):
        optimizer.zero_grad()
        latent, reconstruction = model(train_inputs)
        loss = criterion(reconstruction, targets)
        if loss_start is None:
            loss_start = float(loss.detach().item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        latent, reconstruction = model(train_inputs)
        final_loss = float(criterion(reconstruction, targets).item())
        latent_std = float(latent.std(dim=0).mean().item())
        latent_abs_mean = float(latent.abs().mean().item())

    return {
        'latent_dim': latent_dim,
        'loss_start': round(float(loss_start), 8),
        'final_loss': round(final_loss, 8),
        'latent_shape': list(latent.shape),
        'reconstruction_shape': list(reconstruction.shape),
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'compression_ratio': round(latent_dim / INPUT_DIM, 6),
        'latent_abs_mean': round(latent_abs_mean, 8),
        'latent_std_mean': round(latent_std, 8),
    }


def run() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    clean_inputs, noisy_inputs = _make_dataset()
    raw_noisy_baseline_loss = float(nn.functional.mse_loss(noisy_inputs, clean_inputs).item())

    narrow = _train_autoencoder(
        clean_inputs,
        clean_inputs,
        latent_dim=NARROW_LATENT_DIM,
        seed=SEED,
        steps=1500,
        learning_rate=0.05,
    )
    compression = _train_autoencoder(
        clean_inputs,
        clean_inputs,
        latent_dim=BOTTLENECK_LATENT_DIM,
        seed=SEED,
        steps=1500,
        learning_rate=0.05,
    )
    denoising = _train_autoencoder(
        noisy_inputs,
        clean_inputs,
        latent_dim=BOTTLENECK_LATENT_DIM,
        seed=SEED + 1,
        steps=1800,
        learning_rate=0.05,
    )
    denoising['raw_noisy_baseline_loss'] = round(raw_noisy_baseline_loss, 8)
    denoising['denoising_gain'] = round(raw_noisy_baseline_loss - float(denoising['final_loss']), 8)

    metrics = {
        'device': 'cpu',
        'seed': SEED,
        'input_dim': INPUT_DIM,
        'sample_count': int(clean_inputs.shape[0]),
        'compression_autoencoder': compression,
        'narrow_bottleneck_autoencoder': narrow,
        'denoising_autoencoder': denoising,
        'notes': 'toy reconstruction data only; full-batch deterministic PyTorch autoencoder runs on CPU.',
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
