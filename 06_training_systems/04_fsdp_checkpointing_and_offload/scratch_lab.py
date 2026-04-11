from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'scratch-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
FIGURE_PATH = ARTIFACT_DIR / 'fsdp_memory_tradeoffs.svg'
WORLD_SIZE = 4
COMPONENTS_MB = {
    'parameters_mb': 96.0,
    'gradients_mb': 96.0,
    'optimizer_state_mb': 192.0,
    'activations_mb': 160.0,
}
ACTIVATION_CHECKPOINT_SAVING_RATIO = 0.60
RECOMPUTE_MULTIPLIER = 1.28
CPU_OFFLOAD_TRANSFER_MS = 230
OFFLOAD_GATHER_BUFFER_MB = 12.0


def rounded(value: float) -> float:
    return round(value, 6)


def compute_metrics() -> dict[str, object]:
    param_shard = COMPONENTS_MB['parameters_mb'] / WORLD_SIZE
    grad_shard = COMPONENTS_MB['gradients_mb'] / WORLD_SIZE
    opt_shard = COMPONENTS_MB['optimizer_state_mb'] / WORLD_SIZE
    activation = COMPONENTS_MB['activations_mb']
    checkpointed_activation = activation * (1.0 - ACTIVATION_CHECKPOINT_SAVING_RATIO)

    ddp_full_replica = sum(COMPONENTS_MB.values())
    fsdp_steady = param_shard + grad_shard + opt_shard + activation
    fsdp_forward_peak = fsdp_steady + (COMPONENTS_MB['parameters_mb'] - param_shard)
    checkpointed_steady = param_shard + grad_shard + opt_shard + checkpointed_activation
    checkpointed_peak = checkpointed_steady + (COMPONENTS_MB['parameters_mb'] - param_shard)
    cpu_offload_peak = checkpointed_peak - opt_shard + OFFLOAD_GATHER_BUFFER_MB

    return {
        'cpu_safe_simulation': True,
        'world_size': WORLD_SIZE,
        'sharding_strategy': 'FULL_SHARD',
        'component_mb': COMPONENTS_MB,
        'per_rank_shards_mb': {
            'parameter_shard_mb': rounded(param_shard),
            'gradient_shard_mb': rounded(grad_shard),
            'optimizer_state_shard_mb': rounded(opt_shard),
        },
        'ddp_full_replica_per_rank_mb': rounded(ddp_full_replica),
        'fsdp_steady_gpu_mb': rounded(fsdp_steady),
        'fsdp_forward_peak_gpu_mb': rounded(fsdp_forward_peak),
        'fsdp_checkpointed_activation_mb': rounded(checkpointed_activation),
        'fsdp_checkpointed_peak_gpu_mb': rounded(checkpointed_peak),
        'activation_checkpoint_saving_ratio': rounded(ACTIVATION_CHECKPOINT_SAVING_RATIO),
        'checkpoint_recompute_multiplier': rounded(RECOMPUTE_MULTIPLIER),
        'cpu_offload_gpu_peak_mb': rounded(cpu_offload_peak),
        'cpu_offload_transfer_ms': CPU_OFFLOAD_TRANSFER_MS,
        'memory_saving_vs_ddp_without_offload': rounded(1.0 - checkpointed_peak / ddp_full_replica),
        'memory_saving_vs_ddp_with_cpu_offload': rounded(1.0 - cpu_offload_peak / ddp_full_replica),
        'estimated_step_time_ms': {
            'no_checkpoint_no_offload': 1000,
            'activation_checkpoint_no_offload': int(1000 * RECOMPUTE_MULTIPLIER),
            'activation_checkpoint_cpu_offload': int(1000 * RECOMPUTE_MULTIPLIER + CPU_OFFLOAD_TRANSFER_MS),
        },
        'lifecycle_events': [
            'rank_holds_parameter_shard',
            'all_gather_full_params',
            'forward_compute',
            'drop_full_param_view',
            'recompute_checkpointed_activations',
            'reduce_scatter_gradients',
            'optimizer_step_on_shard',
            'reshard_params',
        ],
        'figure_path': str(FIGURE_PATH.relative_to(UNIT_ROOT)),
    }


def save_svg(metrics: dict[str, object]) -> None:
    values = {
        'DDP replica': metrics['ddp_full_replica_per_rank_mb'],
        'FSDP peak': metrics['fsdp_forward_peak_gpu_mb'],
        'FSDP + ckpt': metrics['fsdp_checkpointed_peak_gpu_mb'],
        'ckpt + offload': metrics['cpu_offload_gpu_peak_mb'],
    }
    width, height = 720, 300
    left, baseline, bar_w = 70, 240, 92
    max_value = max(float(v) for v in values.values())
    bars: list[str] = []
    for idx, (label, value) in enumerate(values.items()):
        height_px = float(value) / max_value * 170
        x = left + idx * 155
        y = baseline - height_px
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{height_px:.2f}" fill="#0f766e" />')
        bars.append(f'<text x="{x}" y="{baseline + 18}" font-size="12">{label}</text>')
        bars.append(f'<text x="{x}" y="{y - 8:.2f}" font-size="12">{value} MB</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#fff" />
  <text x="24" y="32" font-size="18">CPU-safe FSDP memory/offload simulation</text>
  <line x1="50" y1="{baseline}" x2="680" y2="{baseline}" stroke="#111" />
  {''.join(bars)}
</svg>
'''
    FIGURE_PATH.write_text(svg, encoding='utf-8')


def run() -> None:
    metrics = compute_metrics()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_svg(metrics)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
