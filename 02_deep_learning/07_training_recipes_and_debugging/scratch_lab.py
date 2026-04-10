from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'recipe_comparison.svg'

SEED = 7
POLY_DEGREE = 6
EPOCHS = 60
EPSILON = 1e-12
TRAIN_X = np.linspace(-3.0, 3.0, 12, dtype=np.float64)
VAL_X = np.linspace(-3.0, 3.0, 41, dtype=np.float64)
TRAIN_NOISE = np.array(
    [0.0, 0.05, -0.03, 0.02, -0.02, 0.03, -0.04, 0.04, -0.01, 0.02, -0.03, 0.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RecipeConfig:
    name: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    scheduler: str
    epochs: int = EPOCHS
    interpretation: str = ''


def target_function(values: np.ndarray) -> np.ndarray:
    return (0.15 * values**3) - (0.4 * values) + (0.3 * np.sin(1.8 * values))


TRAIN_Y = target_function(TRAIN_X) + TRAIN_NOISE
VAL_Y = target_function(VAL_X)
TRAIN_MEAN: np.ndarray | None = None
TRAIN_STD: np.ndarray | None = None


def _round_float(value: float) -> float:
    return round(float(value), 6)


def rounded_list(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [_round_float(value) for value in array.tolist()]


def make_polynomial_features(values: np.ndarray, degree: int = POLY_DEGREE) -> np.ndarray:
    features = np.stack([values**power for power in range(degree + 1)], axis=1)
    global TRAIN_MEAN, TRAIN_STD
    if TRAIN_MEAN is None or TRAIN_STD is None:
        TRAIN_MEAN = features.mean(axis=0)
        TRAIN_STD = features.std(axis=0)
        TRAIN_STD[TRAIN_STD < 1e-8] = 1.0
    normalized = (features - TRAIN_MEAN) / TRAIN_STD
    normalized[:, 0] = 1.0
    return normalized.astype(np.float64)


TRAIN_FEATURES = make_polynomial_features(TRAIN_X)
VAL_FEATURES = make_polynomial_features(VAL_X)


def schedule_learning_rate(base_lr: float, scheduler: str, epoch: int, total_epochs: int) -> float:
    if scheduler == 'cosine':
        progress = epoch / max(1, total_epochs - 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler == 'linear':
        progress = epoch / max(1, total_epochs - 1)
        return base_lr * (1.0 - (0.8 * progress))
    return base_lr


def classify_recipe(final_train_loss: float, final_val_loss: float, alerts: list[str], gap: float) -> str:
    if any(alert in {'diverged', 'nan'} for alert in alerts):
        return 'diverged'
    if final_train_loss < 0.01 and gap > 0.02:
        return 'overfit_warning'
    if final_train_loss > 0.01:
        return 'underfit_warning'
    return 'stable'


def train_recipe(
    config: RecipeConfig,
    *,
    train_targets: np.ndarray = TRAIN_Y,
    train_features: np.ndarray = TRAIN_FEATURES,
    val_features: np.ndarray = VAL_FEATURES,
    val_targets: np.ndarray = VAL_Y,
    seed_offset: int = 0,
) -> dict[str, object]:
    rng = np.random.default_rng(SEED + seed_offset)
    weights = np.zeros(train_features.shape[1], dtype=np.float64)
    bias = 0.0

    train_history: list[float] = []
    val_history: list[float] = []
    lr_history: list[float] = []
    grad_history: list[float] = []
    alerts: list[str] = []
    first_bad_epoch: int | None = None

    for epoch in range(config.epochs):
        indices = np.arange(train_features.shape[0])
        rng.shuffle(indices)
        current_lr = schedule_learning_rate(
            config.learning_rate,
            config.scheduler,
            epoch,
            config.epochs,
        )
        lr_history.append(current_lr)

        hard_stop = False
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            batch_x = train_features[batch_indices]
            batch_y = train_targets[batch_indices]

            predictions = (batch_x @ weights) + bias
            errors = predictions - batch_y
            grad_w = (batch_x.T @ errors) / len(batch_indices)
            grad_w += config.weight_decay * weights
            grad_b = float(errors.mean())
            grad_norm = float(np.linalg.norm(grad_w))
            grad_history.append(grad_norm)

            if not np.isfinite(grad_norm):
                alerts.append('nan')
                first_bad_epoch = epoch
                hard_stop = True
                break
            if grad_norm > 1e4:
                alerts.append('grad_explosion')
                first_bad_epoch = epoch
                hard_stop = True
                break

            weights -= current_lr * grad_w
            bias -= current_lr * grad_b

            if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
                alerts.append('nan')
                first_bad_epoch = epoch
                hard_stop = True
                break

        train_predictions = (train_features @ weights) + bias
        val_predictions = (val_features @ weights) + bias
        train_loss = 0.5 * float(np.mean((train_predictions - train_targets) ** 2))
        val_loss = 0.5 * float(np.mean((val_predictions - val_targets) ** 2))
        train_history.append(train_loss)
        val_history.append(val_loss)

        if hard_stop or not np.isfinite(train_loss) or not np.isfinite(val_loss):
            alerts.append('diverged')
            if first_bad_epoch is None:
                first_bad_epoch = epoch
            break
        if train_loss > 1e8 or val_loss > 1e8:
            alerts.append('diverged')
            first_bad_epoch = epoch
            break

    final_train_loss = train_history[-1]
    final_val_loss = val_history[-1]
    generalization_gap = final_val_loss - final_train_loss

    return {
        'learning_rate': _round_float(config.learning_rate),
        'batch_size': int(config.batch_size),
        'weight_decay': _round_float(config.weight_decay),
        'scheduler': config.scheduler,
        'epochs_requested': int(config.epochs),
        'epochs_ran': len(train_history),
        'final_train_loss': _round_float(final_train_loss),
        'final_val_loss': _round_float(final_val_loss),
        'best_val_loss': _round_float(min(val_history)),
        'best_val_epoch': int(np.argmin(val_history)),
        'generalization_gap': _round_float(generalization_gap),
        'max_grad_norm': _round_float(max(grad_history) if grad_history else 0.0),
        'final_learning_rate': _round_float(lr_history[-1] if lr_history else config.learning_rate),
        'train_loss_history': rounded_list(train_history),
        'val_loss_history': rounded_list(val_history),
        'log10_val_loss_history': rounded_list(np.log10(np.minimum(np.asarray(val_history), 1e8) + EPSILON)),
        'learning_rate_history': rounded_list(lr_history),
        'alerts': alerts,
        'first_bad_epoch': first_bad_epoch,
        'status': classify_recipe(final_train_loss, final_val_loss, alerts, generalization_gap),
        'interpretation': config.interpretation,
    }


def run_single_batch_overfit() -> dict[str, object]:
    subset_features = TRAIN_FEATURES[:4]
    subset_targets = TRAIN_Y[:4]
    recipe = RecipeConfig(
        name='single_batch_overfit',
        learning_rate=0.1,
        batch_size=4,
        weight_decay=0.0,
        scheduler='cosine',
        epochs=300,
        interpretation='sanity check: 한 배치를 거의 외울 수 있어야 한다.',
    )
    result = train_recipe(
        recipe,
        train_targets=subset_targets,
        train_features=subset_features,
        val_features=subset_features,
        val_targets=subset_targets,
        seed_offset=99,
    )
    return {
        'final_loss': result['final_train_loss'],
        'passed': float(result['final_train_loss']) < 0.001,
        'epochs_ran': result['epochs_ran'],
    }


def run_tiny_subset_replay() -> dict[str, object]:
    subset_features = TRAIN_FEATURES[:6]
    subset_targets = TRAIN_Y[:6]
    recipe = RecipeConfig(
        name='tiny_subset_replay',
        learning_rate=0.08,
        batch_size=3,
        weight_decay=0.0,
        scheduler='cosine',
        epochs=120,
        interpretation='sanity check: 아주 작은 subset에서는 loss가 안정적으로 내려가야 한다.',
    )
    result = train_recipe(
        recipe,
        train_targets=subset_targets,
        train_features=subset_features,
        val_features=subset_features,
        val_targets=subset_targets,
        seed_offset=123,
    )
    return {
        'final_loss': result['final_train_loss'],
        'passed': float(result['final_train_loss']) < 0.01,
        'best_epoch': result['best_val_epoch'],
    }


def polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_text}" />'


def save_svg(series: dict[str, list[float]]) -> None:
    width, height = 780, 460
    left, right = 70, 700
    top, bottom = 60, 360
    max_len = max(len(values) for values in series.values())
    x_min, x_max = 0, max(1, max_len - 1)
    all_values = [value for values in series.values() for value in values]
    y_min = min(all_values) - 0.15
    y_max = max(all_values) + 0.15

    def map_x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * (right - left)

    def map_y(value: float) -> float:
        return bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)

    colors = {
        'small_batch_baseline': '#1c7ed6',
        'large_batch_constant_lr': '#f08c00',
        'weight_decay_scheduler': '#2b8a3e',
        'high_lr_divergence': '#d94841',
    }
    legends = {
        'small_batch_baseline': 'small batch / constant lr',
        'large_batch_constant_lr': 'large batch / constant lr',
        'weight_decay_scheduler': 'weight decay + cosine',
        'high_lr_divergence': 'too-large lr probe',
    }

    lines: list[str] = []
    legend_items: list[str] = []
    legend_y = 84
    for key, values in series.items():
        points = [(map_x(float(step)), map_y(float(loss))) for step, loss in enumerate(values)]
        lines.append(polyline(points, colors[key]))
        legend_items.append(
            f'<rect x="{right - 190}" y="{legend_y - 10}" width="12" height="12" fill="{colors[key]}" />'
            f'<text x="{right - 170}" y="{legend_y}" font-size="13" font-family="Arial, sans-serif">{legends[key]}</text>'
        )
        legend_y += 24

    grid_lines = []
    for step in range(max_len):
        x_value = map_x(float(step))
        grid_lines.append(
            f'<line x1="{x_value}" y1="{top}" x2="{x_value}" y2="{bottom}" stroke="#edf2f7" stroke-width="1" />'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="28" font-size="20" font-family="Arial, sans-serif">Training recipe comparison (scratch)</text>
  <text x="{left}" y="48" font-size="13" font-family="Arial, sans-serif" fill="#495057">y-axis = log10(validation loss), same tiny supervised dataset</text>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2" />
  {''.join(lines)}
  {''.join(legend_items)}
  <text x="{left}" y="402" font-size="13" font-family="Arial, sans-serif" fill="#495057">blue: small-batch baseline is fast but mildly overfits</text>
  <text x="{left}" y="422" font-size="13" font-family="Arial, sans-serif" fill="#495057">green: scheduler + weight decay ends with a slightly smaller validation loss</text>
  <text x="{left}" y="442" font-size="13" font-family="Arial, sans-serif" fill="#495057">red: overly large learning rate quickly diverges</text>
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    recipes = [
        RecipeConfig(
            name='small_batch_baseline',
            learning_rate=0.08,
            batch_size=3,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='작은 batch와 고정 learning rate는 빠르게 loss를 낮추지만 validation gap이 남는다.',
        ),
        RecipeConfig(
            name='large_batch_constant_lr',
            learning_rate=0.08,
            batch_size=12,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='full batch는 더 매끈하지만 같은 epoch budget에서 train loss가 덜 내려간다.',
        ),
        RecipeConfig(
            name='weight_decay_scheduler',
            learning_rate=0.08,
            batch_size=3,
            weight_decay=0.02,
            scheduler='cosine',
            interpretation='weight decay와 cosine decay를 같이 쓰면 late-stage validation loss가 조금 더 안정적이다.',
        ),
        RecipeConfig(
            name='high_lr_divergence',
            learning_rate=1.0,
            batch_size=3,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='너무 큰 learning rate는 gradient explosion과 divergence를 만든다.',
        ),
    ]

    recipe_results = {
        recipe.name: train_recipe(recipe)
        for recipe in recipes
    }
    shifted_label_recipe = RecipeConfig(
        name='shifted_label_bug',
        learning_rate=0.08,
        batch_size=3,
        weight_decay=0.0,
        scheduler='constant',
        interpretation='라벨이 한 칸 밀리면 train/validation gap이 비정상적으로 커진다.',
    )
    shifted_result = train_recipe(
        shifted_label_recipe,
        train_targets=np.roll(TRAIN_Y, 1),
        seed_offset=33,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg({
        name: recipe_results[name]['log10_val_loss_history']
        for name in (
            'small_batch_baseline',
            'large_batch_constant_lr',
            'weight_decay_scheduler',
            'high_lr_divergence',
        )
    })

    baseline = recipe_results['small_batch_baseline']
    regularized = recipe_results['weight_decay_scheduler']
    large_batch = recipe_results['large_batch_constant_lr']
    high_lr = recipe_results['high_lr_divergence']
    single_batch = run_single_batch_overfit()
    tiny_subset = run_tiny_subset_replay()

    metrics = {
        'seed': SEED,
        'train_sample_count': int(len(TRAIN_X)),
        'val_sample_count': int(len(VAL_X)),
        'feature_degree': POLY_DEGREE,
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
        'recipes': recipe_results,
        'debug_probes': {
            'shifted_label_bug': shifted_result,
        },
        'sanity_checks': {
            'single_batch_overfit_final_loss': single_batch['final_loss'],
            'single_batch_overfit_passed': single_batch['passed'],
            'tiny_subset_replay_final_loss': tiny_subset['final_loss'],
            'tiny_subset_replay_passed': tiny_subset['passed'],
            'shifted_label_val_loss': shifted_result['final_val_loss'],
            'label_bug_detected': float(shifted_result['final_val_loss']) > (float(baseline['final_val_loss']) * 10.0),
            'high_lr_first_bad_epoch': high_lr['first_bad_epoch'],
            'high_lr_detected': 'diverged' in high_lr['alerts'],
        },
        'takeaways': {
            'large_batch_minus_small_batch_train_loss': _round_float(
                float(large_batch['final_train_loss']) - float(baseline['final_train_loss'])
            ),
            'regularized_better_than_baseline_on_final_val': float(regularized['final_val_loss']) < float(baseline['final_val_loss']),
            'baseline_overfit_gap': baseline['generalization_gap'],
            'regularized_gap': regularized['generalization_gap'],
            'high_lr_status': high_lr['status'],
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
