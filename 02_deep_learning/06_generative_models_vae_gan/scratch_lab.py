from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'vae_gan_diagnostics.svg'

REAL_POINTS = [
    (-1.2, -0.8),
    (-0.8, -1.1),
    (-1.1, 1.0),
    (-0.7, 0.9),
    (0.9, -1.0),
    (1.1, -0.8),
    (0.8, 1.2),
    (1.2, 0.9),
]
MODE_CENTERS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
VAE_EPSILONS = [
    (0.15, -0.10),
    (-0.20, 0.12),
    (0.18, 0.05),
    (-0.08, -0.15),
    (0.12, -0.18),
    (-0.16, -0.04),
    (0.05, 0.20),
    (-0.11, 0.14),
]
PRIOR_ZS = [(-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (1.0, 1.0), (0.4, -0.6)]
GAN_NOISES = [
    (-1.2, -1.1),
    (-1.0, 0.9),
    (1.1, -1.0),
    (1.2, 1.1),
    (-0.7, -0.8),
    (-0.9, 1.3),
    (0.8, -0.6),
    (0.7, 0.8),
]
INTERPOLATION_STEPS = 5


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _round_point(point: tuple[float, float]) -> list[float]:
    return [_round_float(point[0]), _round_float(point[1])]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _mse(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    total = 0.0
    count = 0
    for left_point, right_point in zip(left, right):
        for left_value, right_value in zip(left_point, right_point):
            total += (left_value - right_value) ** 2
            count += 1
    return total / count


def _pairwise_distance_mean(points: list[tuple[float, float]]) -> float:
    distances: list[float] = []
    for index, point in enumerate(points):
        for other in points[index + 1 :]:
            distances.append(_distance(point, other))
    return _mean(distances)


def _mean_abs(points: list[tuple[float, float]]) -> float:
    values = [abs(value) for point in points for value in point]
    return _mean(values)


def _kl_term(mus: list[tuple[float, float]], logvars: list[tuple[float, float]]) -> float:
    values = []
    for mu, logvar in zip(mus, logvars):
        point_kl = 0.0
        for mu_value, logvar_value in zip(mu, logvar):
            point_kl += math.exp(logvar_value) + (mu_value**2) - 1.0 - logvar_value
        values.append(0.5 * point_kl)
    return _mean(values)


def _encode_mu(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (0.78 * x + 0.18 * y, 0.18 * x + 0.78 * y)


def _encode_logvar(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (-1.9 + (0.05 * x), -1.8 + (0.05 * y))


def _sample_z(mu: tuple[float, float], logvar: tuple[float, float], epsilon: tuple[float, float]) -> tuple[float, float]:
    std = (math.exp(0.5 * logvar[0]), math.exp(0.5 * logvar[1]))
    return (mu[0] + (std[0] * epsilon[0]), mu[1] + (std[1] * epsilon[1]))


def _decode(z: tuple[float, float]) -> tuple[float, float]:
    return (
        math.tanh((0.92 * z[0]) + (0.12 * z[1])),
        math.tanh((0.12 * z[0]) + (0.92 * z[1])),
    )


def _global_mean(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        _mean([point[0] for point in points]),
        _mean([point[1] for point in points]),
    )


def _mode_index(point: tuple[float, float]) -> int:
    distances = [_distance(point, center) for center in MODE_CENTERS]
    return min(range(len(distances)), key=distances.__getitem__)


def _mode_coverage(points: list[tuple[float, float]]) -> int:
    return len({_mode_index(point) for point in points})


def _balanced_generator(noise: tuple[float, float]) -> tuple[float, float]:
    x_sign = 1.0 if noise[0] >= 0.0 else -1.0
    y_sign = 1.0 if noise[1] >= 0.0 else -1.0
    return (
        (1.02 * x_sign) + (0.12 * math.tanh(noise[1])),
        (1.02 * y_sign) + (0.12 * math.tanh(noise[0])),
    )


def _collapsed_generator(noise: tuple[float, float]) -> tuple[float, float]:
    combined = noise[0] + noise[1]
    return (
        1.02 + (0.03 * math.tanh(combined)),
        0.98 + (0.02 * math.tanh(noise[0] - noise[1])),
    )


def _discriminator_logit(point: tuple[float, float]) -> float:
    min_distance = min(_distance(point, center) for center in MODE_CENTERS)
    return 2.4 - (3.0 * min_distance)


def _softplus(value: float) -> float:
    if value > 20:
        return value
    return math.log1p(math.exp(value))


def _generator_loss(points: list[tuple[float, float]]) -> float:
    return _mean([_softplus(-_discriminator_logit(point)) for point in points])


def _discriminator_loss(real_points: list[tuple[float, float]], fake_points: list[tuple[float, float]]) -> float:
    real_term = [_softplus(-_discriminator_logit(point)) for point in real_points]
    fake_term = [_softplus(_discriminator_logit(point)) for point in fake_points]
    return _mean(real_term + fake_term)


def _interpolate(start: tuple[float, float], end: tuple[float, float], steps: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(steps):
        alpha = index / (steps - 1)
        z = (
            ((1.0 - alpha) * start[0]) + (alpha * end[0]),
            ((1.0 - alpha) * start[1]) + (alpha * end[1]),
        )
        points.append(_decode(z))
    return points


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(left, right) for left, right in zip(points, points[1:]))


def _to_svg_point(point: tuple[float, float], x_offset: float) -> tuple[float, float]:
    scale = 80.0
    x = x_offset + 150.0 + (point[0] * scale)
    y = 240.0 - (point[1] * scale)
    return (x, y)


def _render_panel(
    *,
    title: str,
    x_offset: float,
    base_points: list[tuple[float, float]],
    overlays: list[tuple[list[tuple[float, float]], str, str]],
    notes: list[str],
) -> str:
    parts = [
        f'<rect x="{x_offset + 20:.0f}" y="36" width="280" height="320" rx="14" fill="#ffffff" stroke="#cbd5e1" />',
        f'<text x="{x_offset + 40:.0f}" y="68" font-size="20" font-weight="bold" fill="#0f172a">{escape(title)}</text>',
    ]
    for center in MODE_CENTERS:
        x, y = _to_svg_point(center, x_offset)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#e2e8f0" stroke="#94a3b8" />')
    for point in base_points:
        x, y = _to_svg_point(point, x_offset)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#111827" />')
    for index, (points, color, label) in enumerate(overlays):
        if len(points) > 1:
            polyline = ' '.join(
                f'{_to_svg_point(point, x_offset)[0]:.2f},{_to_svg_point(point, x_offset)[1]:.2f}'
                for point in points
            )
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />')
        for point in points:
            x, y = _to_svg_point(point, x_offset)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" />')
        legend_y = 300 + (index * 18)
        parts.append(
            f'<rect x="{x_offset + 40:.0f}" y="{legend_y - 10}" width="10" height="10" fill="{color}" />'
            f'<text x="{x_offset + 58:.0f}" y="{legend_y}" font-size="12" fill="#334155">{escape(label)}</text>'
        )
    for index, note in enumerate(notes):
        parts.append(
            f'<text x="{x_offset + 40:.0f}" y="{88 + (index * 18):.0f}" font-size="12" fill="#475569">{escape(note)}</text>'
        )
    return ''.join(parts)


def _render_svg(
    interpolation_points: list[tuple[float, float]],
    prior_samples: list[tuple[float, float]],
    balanced_samples: list[tuple[float, float]],
    collapsed_samples: list[tuple[float, float]],
) -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="400" viewBox="0 0 940 400">'
        '<rect width="100%" height="100%" fill="#f8fafc" />'
        '<text x="28" y="30" font-size="26" font-weight="bold" fill="#0f172a">VAE vs GAN diagnostics</text>'
        '<text x="28" y="52" font-size="13" fill="#475569">'
        '왼쪽은 VAE latent interpolation / prior sampling, 오른쪽은 GAN balanced vs collapsed sample coverage를 보여 준다.</text>'
        + _render_panel(
            title='VAE latent sampling',
            x_offset=0.0,
            base_points=REAL_POINTS,
            overlays=[
                (interpolation_points, '#2563eb', 'decoded interpolation'),
                (prior_samples, '#16a34a', 'prior samples'),
            ],
            notes=[
                '검은 점: real toy data',
                '파란 선: latent interpolation',
                '초록 점: prior z decode',
            ],
        )
        + _render_panel(
            title='GAN mode coverage',
            x_offset=320.0,
            base_points=REAL_POINTS,
            overlays=[
                (balanced_samples, '#f97316', 'balanced generator'),
                (collapsed_samples, '#dc2626', 'collapsed generator'),
            ],
            notes=[
                '주황 점: 여러 mode를 덮는 생성기',
                '빨간 점: 한 mode로 몰리는 collapse',
                'loss만 보면 둘 다 그럴듯해 보일 수 있다',
            ],
        )
        + '</svg>'
    )
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    healthy_mu = [_encode_mu(point) for point in REAL_POINTS]
    healthy_logvar = [_encode_logvar(point) for point in REAL_POINTS]
    healthy_z = [_sample_z(mu, logvar, epsilon) for mu, logvar, epsilon in zip(healthy_mu, healthy_logvar, VAE_EPSILONS)]
    healthy_reconstruction = [_decode(point) for point in healthy_z]
    healthy_recon_mse = _mse(healthy_reconstruction, REAL_POINTS)
    healthy_kl = _kl_term(healthy_mu, healthy_logvar)

    interpolation_points = _interpolate(healthy_mu[0], healthy_mu[-1], INTERPOLATION_STEPS)
    prior_samples = [_decode(point) for point in PRIOR_ZS]

    collapsed_mu = [(0.0, 0.0) for _ in REAL_POINTS]
    collapsed_logvar = [(0.0, 0.0) for _ in REAL_POINTS]
    mean_point = _global_mean(REAL_POINTS)
    collapsed_reconstruction = [mean_point for _ in REAL_POINTS]
    collapsed_recon_mse = _mse(collapsed_reconstruction, REAL_POINTS)
    collapsed_kl = _kl_term(collapsed_mu, collapsed_logvar)

    balanced_samples = [_balanced_generator(noise) for noise in GAN_NOISES]
    collapsed_samples = [_collapsed_generator(noise) for noise in GAN_NOISES]
    balanced_g_loss = _generator_loss(balanced_samples)
    balanced_d_loss = _discriminator_loss(REAL_POINTS, balanced_samples)
    collapsed_g_loss = _generator_loss(collapsed_samples)
    collapsed_d_loss = _discriminator_loss(REAL_POINTS, collapsed_samples)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _render_svg(interpolation_points, prior_samples, balanced_samples, collapsed_samples)

    metrics = {
        'seed': 0,
        'dataset_point_count': len(REAL_POINTS),
        'vae': {
            'input_dim': 2,
            'latent_dim': 2,
            'mu_shape': [len(healthy_mu), 2],
            'logvar_shape': [len(healthy_logvar), 2],
            'sampled_latent_shape': [len(healthy_z), 2],
            'reconstruction_shape': [len(healthy_reconstruction), 2],
            'reconstruction_mse': _round_float(healthy_recon_mse),
            'kl_term': _round_float(healthy_kl),
            'interpolation_steps': INTERPOLATION_STEPS,
            'interpolation_path_length': _round_float(_path_length(interpolation_points)),
            'prior_sample_count': len(prior_samples),
            'prior_sample_spread': _round_float(_pairwise_distance_mean(prior_samples)),
            'reparameterization_example': {
                'mu': _round_point(healthy_mu[0]),
                'epsilon': _round_point(VAE_EPSILONS[0]),
                'sampled_z': _round_point(healthy_z[0]),
            },
            'posterior_collapse_probe': {
                'healthy_latent_usage': _round_float(_mean_abs(healthy_mu)),
                'collapsed_latent_usage': _round_float(_mean_abs(collapsed_mu)),
                'healthy_reconstruction_mse': _round_float(healthy_recon_mse),
                'collapsed_reconstruction_mse': _round_float(collapsed_recon_mse),
                'healthy_kl_term': _round_float(healthy_kl),
                'collapsed_kl_term': _round_float(collapsed_kl),
                'collapse_detected': (_mean_abs(collapsed_mu) < _mean_abs(healthy_mu)) and (collapsed_recon_mse > healthy_recon_mse),
            },
        },
        'gan': {
            'noise_dim': 2,
            'generated_shape': [len(balanced_samples), 2],
            'balanced_generator_loss': _round_float(balanced_g_loss),
            'balanced_discriminator_loss': _round_float(balanced_d_loss),
            'balanced_mode_coverage': _mode_coverage(balanced_samples),
            'balanced_pairwise_distance_mean': _round_float(_pairwise_distance_mean(balanced_samples)),
            'collapsed_generator_loss': _round_float(collapsed_g_loss),
            'collapsed_discriminator_loss': _round_float(collapsed_d_loss),
            'collapsed_mode_coverage': _mode_coverage(collapsed_samples),
            'collapsed_pairwise_distance_mean': _round_float(_pairwise_distance_mean(collapsed_samples)),
            'discriminator_real_mean': _round_float(_mean([_discriminator_logit(point) for point in REAL_POINTS])),
            'discriminator_balanced_fake_mean': _round_float(_mean([_discriminator_logit(point) for point in balanced_samples])),
            'discriminator_collapsed_fake_mean': _round_float(_mean([_discriminator_logit(point) for point in collapsed_samples])),
            'loss_only_is_ambiguous': abs(collapsed_g_loss - balanced_g_loss) < 0.2,
            'collapse_detected': _mode_coverage(collapsed_samples) < _mode_coverage(balanced_samples),
        },
        'contrast': {
            'vae_strength': 'smooth latent interpolation and explicit sampling geometry',
            'gan_strength': 'sharper samples near the target modes without a reconstruction term',
            'vae_tradeoff': 'posterior collapse can make z almost unused',
            'gan_tradeoff': 'mode collapse can look plausible while diversity disappears',
        },
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
