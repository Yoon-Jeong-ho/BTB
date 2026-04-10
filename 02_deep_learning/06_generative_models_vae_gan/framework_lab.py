from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
DEVICE = 'cpu'
SEED = 11

REAL_POINTS = torch.tensor(
    [
        [-1.2, -0.8],
        [-0.8, -1.1],
        [-1.1, 1.0],
        [-0.7, 0.9],
        [0.9, -1.0],
        [1.1, -0.8],
        [0.8, 1.2],
        [1.2, 0.9],
    ],
    dtype=torch.float32,
)
MODE_CENTERS = torch.tensor(
    [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)
VAE_EPS = torch.tensor(
    [
        [0.15, -0.10],
        [-0.20, 0.12],
        [0.18, 0.05],
        [-0.08, -0.15],
        [0.12, -0.18],
        [-0.16, -0.04],
        [0.05, 0.20],
        [-0.11, 0.14],
    ],
    dtype=torch.float32,
)
GAN_NOISE = torch.tensor(
    [
        [-1.2, -1.1],
        [-1.0, 0.9],
        [1.1, -1.0],
        [1.2, 1.1],
        [-0.7, -0.8],
        [-0.9, 1.3],
        [0.8, -0.6],
        [0.7, 0.8],
    ],
    dtype=torch.float32,
)
PRIOR_Z = torch.tensor(
    [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 0.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [0.4, -0.6],
    ],
    dtype=torch.float32,
)
INTERPOLATION_STEPS = 5


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _rounded_list(tensor: torch.Tensor) -> list[float]:
    return [_round_float(value) for value in tensor.detach().cpu().view(-1).tolist()]


def _pairwise_distance_mean(points: torch.Tensor) -> float:
    if points.size(0) < 2:
        return 0.0
    distances = torch.cdist(points, points, p=2)
    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    selected = distances[mask]
    return float(selected.mean().item())


def _mode_coverage(points: torch.Tensor) -> int:
    distances = torch.cdist(points, MODE_CENTERS, p=2)
    assignments = distances.argmin(dim=1)
    return int(assignments.unique().numel())


class TinyVAE(nn.Module):
    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self.collapsed = collapsed
        self.mu_layer = nn.Linear(2, 2, bias=True)
        self.logvar_layer = nn.Linear(2, 2, bias=True)
        self.decoder = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.mu_layer.weight.copy_(torch.tensor([[0.82, 0.18], [0.18, 0.82]], dtype=torch.float32))
            self.mu_layer.bias.zero_()
            self.logvar_layer.weight.copy_(torch.tensor([[0.05, 0.00], [0.00, 0.05]], dtype=torch.float32))
            self.logvar_layer.bias.copy_(torch.tensor([-1.75, -1.7], dtype=torch.float32))
            self.decoder.weight.copy_(torch.tensor([[0.95, 0.08], [0.08, 0.95]], dtype=torch.float32))
            self.decoder.bias.zero_()

    def forward(self, inputs: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.collapsed:
            mu = torch.zeros(inputs.size(0), 2, dtype=inputs.dtype, device=inputs.device)
            logvar = torch.zeros_like(mu)
            z = eps
            mean_point = inputs.mean(dim=0, keepdim=True)
            reconstruction = mean_point.repeat(inputs.size(0), 1)
            return mu, logvar, z, reconstruction

        mu = torch.tanh(self.mu_layer(inputs))
        logvar = -1.6 + (0.25 * torch.tanh(self.logvar_layer(inputs)))
        std = torch.exp(0.5 * logvar)
        z = mu + (std * eps)
        reconstruction = torch.tanh(self.decoder(z))
        return mu, logvar, z, reconstruction

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.collapsed:
            mean_point = REAL_POINTS.mean(dim=0, keepdim=True)
            return mean_point.repeat(latent.size(0), 1)
        return torch.tanh(self.decoder(latent))


class QuadrantGenerator(nn.Module):
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x_sign = torch.where(noise[:, :1] >= 0.0, torch.ones_like(noise[:, :1]), -torch.ones_like(noise[:, :1]))
        y_sign = torch.where(noise[:, 1:] >= 0.0, torch.ones_like(noise[:, 1:]), -torch.ones_like(noise[:, 1:]))
        return torch.cat(
            [
                (1.02 * x_sign) + (0.12 * torch.tanh(noise[:, 1:])),
                (1.02 * y_sign) + (0.12 * torch.tanh(noise[:, :1])),
            ],
            dim=1,
        )


class CollapsedGenerator(nn.Module):
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        combined = noise[:, :1] + noise[:, 1:]
        return torch.cat(
            [
                1.02 + (0.03 * torch.tanh(combined)),
                0.98 + (0.02 * torch.tanh(noise[:, :1] - noise[:, 1:])),
            ],
            dim=1,
        )


class ModeDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('modes', MODE_CENTERS.clone())

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(points, self.modes, p=2)
        min_distance = distances.min(dim=1).values
        return 2.4 - (3.0 * min_distance)


def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + (mu**2) - 1.0 - logvar, dim=1))


def _generator_loss(logits: torch.Tensor) -> float:
    return float(F.softplus(-logits).mean().item())


def _discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
    return float((F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()).item())


def run() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    data = REAL_POINTS.to(DEVICE)
    eps = VAE_EPS.to(DEVICE)
    healthy_vae = TinyVAE(collapsed=False).to(DEVICE)
    collapsed_vae = TinyVAE(collapsed=True).to(DEVICE)

    with torch.no_grad():
        mu, logvar, latent, reconstruction = healthy_vae(data, eps)
        vae_recon = float(F.mse_loss(reconstruction, data).item())
        vae_kl = float(_kl(mu, logvar).item())
        posterior_usage_mean_abs = float(mu.abs().mean().item())
        prior_samples = healthy_vae.decode(PRIOR_Z.to(DEVICE))
        interpolation_latent = torch.stack(
            [
                (1.0 - alpha) * mu[0] + alpha * mu[-1]
                for alpha in torch.linspace(0.0, 1.0, INTERPOLATION_STEPS)
            ],
            dim=0,
        )
        interpolation = healthy_vae.decode(interpolation_latent)

        collapsed_mu, collapsed_logvar, _, collapsed_reconstruction = collapsed_vae(data, torch.zeros_like(eps))
        collapsed_recon = float(F.mse_loss(collapsed_reconstruction, data).item())
        collapsed_kl = float(_kl(collapsed_mu, collapsed_logvar).item())
        collapsed_usage = float(collapsed_mu.abs().mean().item())

        generator = QuadrantGenerator().to(DEVICE)
        collapsed_generator = CollapsedGenerator().to(DEVICE)
        discriminator = ModeDiscriminator().to(DEVICE)
        gan_samples = generator(GAN_NOISE.to(DEVICE))
        collapsed_samples = collapsed_generator(GAN_NOISE.to(DEVICE))
        real_logits = discriminator(data)
        fake_logits = discriminator(gan_samples)
        collapsed_logits = discriminator(collapsed_samples)

    gan_generator_loss = _generator_loss(fake_logits)
    gan_discriminator_loss = _discriminator_loss(real_logits, fake_logits)
    collapsed_generator_loss = _generator_loss(collapsed_logits)

    metrics = {
        'device': DEVICE,
        'seed': SEED,
        'dataset_point_count': int(data.size(0)),
        'vae': {
            'latent_dim': 2,
            'mu_shape': list(mu.shape),
            'logvar_shape': list(logvar.shape),
            'sampled_latent_shape': list(latent.shape),
            'reconstruction_shape': list(reconstruction.shape),
            'prior_sample_shape': list(prior_samples.shape),
            'interpolation_steps': INTERPOLATION_STEPS,
            'final_reconstruction_loss': _round_float(vae_recon),
            'final_kl_loss': _round_float(vae_kl),
            'posterior_usage_mean_abs': _round_float(posterior_usage_mean_abs),
            'prior_sample_spread': _round_float(_pairwise_distance_mean(prior_samples)),
            'interpolation_path_length': _round_float(_pairwise_distance_mean(interpolation)),
            'reparameterization_example': {
                'mu': _rounded_list(mu[0]),
                'eps': _rounded_list(eps[0]),
                'sampled_z': _rounded_list(latent[0]),
            },
            'collapsed_probe': {
                'collapsed_reconstruction_loss': _round_float(collapsed_recon),
                'collapsed_kl_loss': _round_float(collapsed_kl),
                'collapsed_latent_usage_mean_abs': _round_float(collapsed_usage),
                'collapse_detected': (collapsed_usage < posterior_usage_mean_abs)
                and (collapsed_recon > vae_recon)
                and (collapsed_kl < vae_kl),
            },
        },
        'posterior_usage_mean_abs': _round_float(posterior_usage_mean_abs),
        'gan': {
            'noise_dim': int(GAN_NOISE.size(1)),
            'sample_shape': list(gan_samples.shape),
            'generator_loss': _round_float(gan_generator_loss),
            'discriminator_loss': _round_float(gan_discriminator_loss),
            'mode_coverage': _mode_coverage(gan_samples),
            'pairwise_distance_mean': _round_float(_pairwise_distance_mean(gan_samples)),
            'discriminator_real_mean': _round_float(float(real_logits.mean().item())),
            'discriminator_fake_mean': _round_float(float(fake_logits.mean().item())),
            'loss_only_is_ambiguous': abs(collapsed_generator_loss - gan_generator_loss) < 0.2,
            'collapsed_probe': {
                'mode_coverage': _mode_coverage(collapsed_samples),
                'pairwise_distance_mean': _round_float(_pairwise_distance_mean(collapsed_samples)),
                'generator_loss': _round_float(collapsed_generator_loss),
                'collapse_detected': _mode_coverage(collapsed_samples) < _mode_coverage(gan_samples),
            },
        },
        'notes': 'Tiny PyTorch modules only; deterministic CPU forward passes for VAE sampling and GAN mode-coverage diagnostics.',
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
