from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'autoencoder_bottleneck.svg'

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
BOTTLENECK_DIMS = [1, 2, 3]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


PRIMARY_BASIS = [_normalize(vector) for vector in PRIMARY_BASIS_RAW]
NOISE_BASIS = [_normalize(vector) for vector in NOISE_BASIS_RAW]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [scalar * value for value in vector]


def _mean_squared_error(left: list[list[float]], right: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            total += (left_value - right_value) ** 2
            count += 1
    return total / count


def _round_vector(vector: list[float]) -> list[float]:
    return [round(value, 6) for value in vector]


def _decode_coefficients(coefficients: list[float]) -> list[float]:
    reconstruction = list(MEAN_VECTOR)
    for basis_vector, coefficient in zip(PRIMARY_BASIS, coefficients):
        reconstruction = _add(reconstruction, _scale(basis_vector, coefficient))
    return reconstruction


def _make_clean_samples() -> list[list[float]]:
    clean_samples: list[list[float]] = []
    for coefficients in LATENT_CODES:
        clean_samples.append(_decode_coefficients(coefficients))
    return clean_samples


def _make_noisy_samples(clean_samples: list[list[float]]) -> list[list[float]]:
    noisy_samples: list[list[float]] = []
    for clean_sample, noise_coefficients in zip(clean_samples, NOISE_CODES):
        noise = [0.0 for _ in clean_sample]
        for basis_vector, coefficient in zip(NOISE_BASIS, noise_coefficients):
            noise = _add(noise, _scale(basis_vector, coefficient))
        noisy_samples.append(_add(clean_sample, noise))
    return noisy_samples


def _encode(sample: list[float], latent_dim: int) -> list[float]:
    centered = _subtract(sample, MEAN_VECTOR)
    return [_dot(centered, basis_vector) for basis_vector in PRIMARY_BASIS[:latent_dim]]


def _reconstruct(
    samples: list[list[float]],
    latent_dim: int,
    *,
    targets: list[list[float]] | None = None,
) -> tuple[list[list[float]], list[list[float]], float]:
    latents: list[list[float]] = []
    reconstructions: list[list[float]] = []
    for sample in samples:
        latent = _encode(sample, latent_dim)
        latents.append(latent)
        reconstructions.append(_decode_coefficients(latent))
    target_rows = samples if targets is None else targets
    return latents, reconstructions, _mean_squared_error(reconstructions, target_rows)


def _render_svg(clean_mses: dict[int, float], raw_noisy_mse: float, denoised_mse: float) -> None:
    bar_specs = [
        ('clean latent=1', clean_mses[1], '#ffd43b'),
        ('clean latent=2', clean_mses[2], '#74c0fc'),
        ('clean latent=3', clean_mses[3], '#69db7c'),
        ('raw noisy', raw_noisy_mse, '#ff8787'),
        ('denoised latent=3', denoised_mse, '#9775fa'),
    ]
    max_value = max(value for _, value, _ in bar_specs)
    width = 920
    height = 420
    chart_top = 86
    chart_left = 84
    chart_height = 250
    bar_width = 110
    gap = 36
    labels = []
    bars = []
    value_text = []
    for index, (label, value, color) in enumerate(bar_specs):
        x = chart_left + index * (bar_width + gap)
        usable_height = chart_height * (value / max_value) if max_value > 0 else 0.0
        y = chart_top + chart_height - usable_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{usable_height:.2f}" fill="{color}" stroke="#2d3748" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{chart_top + chart_height + 28}" font-size="13" '
            f'text-anchor="middle" fill="#1a202c">{escape(label)}</text>'
        )
        value_text.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" font-size="13" text-anchor="middle" '
            f'fill="#1a202c">{value:.4f}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fbff" />'
        '<text x="28" y="34" font-size="24" font-weight="bold" fill="#1a202c">'
        'Autoencoder bottleneck diagnostics</text>'
        '<text x="28" y="58" font-size="13" fill="#4a5568">'
        '낮을수록 더 잘 복원하며, denoising bar는 noisy input 대비 clean target MSE를 보여 준다.</text>'
        f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{width - 40}" y2="{chart_top + chart_height}" stroke="#94a3b8" stroke-width="2" />'
        f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#94a3b8" stroke-width="2" />'
        + ''.join(bars)
        + ''.join(labels)
        + ''.join(value_text)
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    clean_samples = _make_clean_samples()
    noisy_samples = _make_noisy_samples(clean_samples)

    clean_results: dict[int, dict[str, object]] = {}
    clean_mses: dict[int, float] = {}
    for latent_dim in BOTTLENECK_DIMS:
        latents, reconstructions, mse = _reconstruct(clean_samples, latent_dim)
        clean_results[latent_dim] = {
            'latent_dim': latent_dim,
            'latent_shape': [len(latents), latent_dim],
            'reconstruction_shape': [len(reconstructions), len(reconstructions[0])],
            'reconstruction_mse': round(mse, 8),
            'first_sample_latent': _round_vector(latents[0]),
            'first_sample_reconstruction': _round_vector(reconstructions[0]),
        }
        clean_mses[latent_dim] = mse

    denoising_latents, denoised_reconstructions, denoised_mse = _reconstruct(noisy_samples, 3, targets=clean_samples)
    raw_noisy_mse = _mean_squared_error(noisy_samples, clean_samples)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _render_svg(clean_mses=clean_mses, raw_noisy_mse=raw_noisy_mse, denoised_mse=denoised_mse)

    metrics = {
        'seed': 0,
        'input_dim': len(clean_samples[0]),
        'sample_count': len(clean_samples),
        'encoder_decoder_roles': {
            'encoder': 'input -> latent coefficients via basis projection',
            'latent': 'compressed code that keeps shared structure',
            'decoder': 'latent -> reconstruction via basis expansion',
        },
        'bottleneck_dims_compared': BOTTLENECK_DIMS,
        'bottleneck_results': {str(key): value for key, value in clean_results.items()},
        'compression_variant': {
            'selected_latent_dim': 3,
            'compression_ratio': round(3 / len(clean_samples[0]), 6),
            'best_reconstruction_mse': round(clean_mses[3], 8),
            'narrow_bottleneck_dim': 1,
            'narrow_vs_selected_mse_gap': round(clean_mses[1] - clean_mses[3], 8),
        },
        'denoising_variant': {
            'noise_level': 0.18,
            'raw_noisy_mse': round(raw_noisy_mse, 8),
            'denoised_mse': round(denoised_mse, 8),
            'denoising_improves_over_noisy_input': denoised_mse < raw_noisy_mse,
            'target': 'clean reconstruction from noisy input',
        },
        'latent_preview': {
            'sample_0': _round_vector(denoising_latents[0]),
            'sample_1': _round_vector(denoising_latents[1]),
        },
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
