from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
DEVICE = 'cpu'
SEED = 7
EPOCHS = 160
POLY_DEGREE = 6


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

TRAIN_X = torch.linspace(-3.0, 3.0, 12, dtype=torch.float32)
VAL_X = torch.linspace(-3.0, 3.0, 41, dtype=torch.float32)
TRAIN_NOISE = torch.tensor(
    [0.0, 0.05, -0.03, 0.02, -0.02, 0.03, -0.04, 0.04, -0.01, 0.02, -0.03, 0.0],
    dtype=torch.float32,
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


def target_function(values: torch.Tensor) -> torch.Tensor:
    return (0.15 * values**3) - (0.4 * values) + (0.3 * torch.sin(1.8 * values))


TRAIN_Y = target_function(TRAIN_X) + TRAIN_NOISE
VAL_Y = target_function(VAL_X)


def _round_float(value: float) -> float:
    return round(float(value), 6)


def rounded_list(values: list[float] | torch.Tensor) -> list[float]:
    if isinstance(values, torch.Tensor):
        return [_round_float(value) for value in values.detach().cpu().view(-1).tolist()]
    return [_round_float(value) for value in values]


def make_polynomial_features(values: torch.Tensor, degree: int = POLY_DEGREE) -> torch.Tensor:
    base = torch.stack([values**power for power in range(degree + 1)], dim=1)
    means = base.mean(dim=0)
    stds = base.std(dim=0, unbiased=False)
    stds = torch.where(stds < 1e-8, torch.ones_like(stds), stds)
    normalized = (base - means) / stds
    normalized[:, 0] = 1.0
    return normalized


TRAIN_FEATURES = make_polynomial_features(TRAIN_X)
VAL_FEATURES = make_polynomial_features(VAL_X)


def make_model() -> torch.nn.Module:
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(TRAIN_FEATURES.size(1), 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 1),
    )


def train_recipe(
    config: RecipeConfig,
    *,
    train_targets: torch.Tensor = TRAIN_Y,
    train_features: torch.Tensor = TRAIN_FEATURES,
    val_targets: torch.Tensor = VAL_Y,
    val_features: torch.Tensor = VAL_FEATURES,
    seed_offset: int = 0,
) -> dict[str, object]:
    model = make_model().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    generator = torch.Generator().manual_seed(SEED + 100 + seed_offset)
    train_history: list[float] = []
    val_history: list[float] = []
    lr_history: list[float] = []
    grad_history: list[float] = []
    alerts: list[str] = []
    first_bad_epoch: int | None = None

    for epoch in range(config.epochs):
        permutation = torch.randperm(train_features.size(0), generator=generator)
        hard_stop = False
        model.train()
        for start in range(0, len(permutation), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            batch_x = train_features[batch_indices].to(DEVICE)
            batch_y = train_targets[batch_indices].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = torch.mean((predictions - batch_y) ** 2) / 2.0
            if not torch.isfinite(loss):
                alerts.append('nan')
                first_bad_epoch = epoch
                hard_stop = True
                break

            loss.backward()
            grad_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq += float(torch.sum(parameter.grad.detach() ** 2).item())
            grad_norm = math.sqrt(grad_sq)
            grad_history.append(grad_norm)
            if grad_norm > 1e4:
                alerts.append('grad_explosion')
                first_bad_epoch = epoch
                hard_stop = True
                break

            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = float((torch.mean((model(train_features) - train_targets.unsqueeze(1)) ** 2) / 2.0).item())
            val_loss = float((torch.mean((model(val_features) - val_targets.unsqueeze(1)) ** 2) / 2.0).item())
        train_history.append(train_loss)
        val_history.append(val_loss)
        lr_history.append(float(optimizer.param_groups[0]['lr']))

        if scheduler is not None:
            scheduler.step()
        if hard_stop or not math.isfinite(train_loss) or not math.isfinite(val_loss):
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
        'optimizer': 'SGD',
        'learning_rate': _round_float(config.learning_rate),
        'batch_size': int(config.batch_size),
        'weight_decay': _round_float(config.weight_decay),
        'scheduler': config.scheduler,
        'epochs_requested': int(config.epochs),
        'epochs_ran': len(train_history),
        'final_train_loss': _round_float(final_train_loss),
        'final_val_loss': _round_float(final_val_loss),
        'best_val_loss': _round_float(min(val_history)),
        'best_val_epoch': int(torch.tensor(val_history).argmin().item()),
        'generalization_gap': _round_float(generalization_gap),
        'max_grad_norm': _round_float(max(grad_history) if grad_history else 0.0),
        'final_learning_rate': _round_float(lr_history[-1] if lr_history else config.learning_rate),
        'train_loss_history': rounded_list(train_history),
        'val_loss_history': rounded_list(val_history),
        'learning_rate_history': rounded_list(lr_history),
        'alerts': alerts,
        'first_bad_epoch': first_bad_epoch,
        'interpretation': config.interpretation,
    }


def run_single_batch_overfit() -> dict[str, object]:
    recipe = RecipeConfig(
        name='single_batch_overfit',
        learning_rate=0.05,
        batch_size=4,
        weight_decay=0.0,
        scheduler='constant',
        epochs=400,
        interpretation='sanity check: 작은 MLP도 네 샘플은 거의 외워야 한다.',
    )
    subset_features = TRAIN_FEATURES[:4]
    subset_targets = TRAIN_Y[:4]
    result = train_recipe(
        recipe,
        train_targets=subset_targets,
        train_features=subset_features,
        val_targets=subset_targets,
        val_features=subset_features,
        seed_offset=88,
    )
    return {
        'final_loss': result['final_train_loss'],
        'passed': float(result['final_train_loss']) < 0.001,
        'epochs_ran': result['epochs_ran'],
    }


def run() -> None:
    recipes = [
        RecipeConfig(
            name='baseline_tiny_mlp',
            learning_rate=0.02,
            batch_size=4,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='tiny MLP도 작은 batch에서는 빠르게 train loss를 내리지만 validation gap이 남는다.',
        ),
        RecipeConfig(
            name='weight_decay_scheduler_tiny_mlp',
            learning_rate=0.02,
            batch_size=4,
            weight_decay=0.01,
            scheduler='cosine',
            interpretation='weight decay + cosine scheduler는 late-stage overfit를 조금 완화한다.',
        ),
        RecipeConfig(
            name='large_batch_tiny_mlp',
            learning_rate=0.02,
            batch_size=12,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='full batch는 gradient noise가 줄지만 같은 epoch budget에서 fit 속도가 둔해진다.',
        ),
        RecipeConfig(
            name='high_lr_tiny_mlp',
            learning_rate=0.6,
            batch_size=4,
            weight_decay=0.0,
            scheduler='constant',
            interpretation='과도한 learning rate는 PyTorch에서도 gradient explosion과 divergence를 만든다.',
        ),
    ]

    recipe_results = {recipe.name: train_recipe(recipe) for recipe in recipes}
    shifted_label_recipe = RecipeConfig(
        name='shifted_label_bug_tiny_mlp',
        learning_rate=0.02,
        batch_size=4,
        weight_decay=0.0,
        scheduler='constant',
        interpretation='라벨 misalignment는 validation loss를 비정상적으로 키운다.',
    )
    shifted_result = train_recipe(
        shifted_label_recipe,
        train_targets=torch.roll(TRAIN_Y, shifts=1),
        seed_offset=44,
    )

    baseline = recipe_results['baseline_tiny_mlp']
    regularized = recipe_results['weight_decay_scheduler_tiny_mlp']
    high_lr = recipe_results['high_lr_tiny_mlp']
    single_batch = run_single_batch_overfit()

    metrics = {
        'seed': SEED,
        'device': DEVICE,
        'model_name': 'tiny_mlp_gelu',
        'feature_degree': POLY_DEGREE,
        'train_sample_count': int(TRAIN_X.numel()),
        'val_sample_count': int(VAL_X.numel()),
        'recipes': recipe_results,
        'debug_probes': {
            'shifted_label_bug_tiny_mlp': shifted_result,
        },
        'sanity_checks': {
            'single_batch_overfit_final_loss': single_batch['final_loss'],
            'single_batch_overfit_passed': single_batch['passed'],
            'shifted_label_val_loss': shifted_result['final_val_loss'],
            'label_bug_detected': float(shifted_result['final_val_loss']) > (float(baseline['final_val_loss']) * 10.0),
            'high_lr_first_bad_epoch': high_lr['first_bad_epoch'],
            'high_lr_detected': 'diverged' in high_lr['alerts'],
        },
        'takeaways': {
            'regularized_better_than_baseline_on_final_val': float(regularized['final_val_loss']) < float(baseline['final_val_loss']),
            'baseline_gap': baseline['generalization_gap'],
            'regularized_gap': regularized['generalization_gap'],
            'high_lr_status': high_lr['alerts'],
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
