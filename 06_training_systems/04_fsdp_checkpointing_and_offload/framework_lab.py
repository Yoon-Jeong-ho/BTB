from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
WORLD_SIZE = 4
TOTALS_MB = {
    'parameters_mb': 96.0,
    'gradients_mb': 96.0,
    'optimizer_state_mb': 192.0,
}
LAYERS = [
    ('embed', 24.0),
    ('block_00', 24.0),
    ('block_01', 32.0),
    ('lm_head', 16.0),
]


def rounded(value: float) -> float:
    return round(value, 6)


def build_rank_shards() -> list[dict[str, object]]:
    param_shard = TOTALS_MB['parameters_mb'] / WORLD_SIZE
    grad_shard = TOTALS_MB['gradients_mb'] / WORLD_SIZE
    opt_shard = TOTALS_MB['optimizer_state_mb'] / WORLD_SIZE
    shards = []
    for rank in range(WORLD_SIZE):
        primary_layer = LAYERS[rank % len(LAYERS)][0]
        shards.append({
            'rank': rank,
            'primary_layer': primary_layer,
            'parameter_shard_mb': rounded(param_shard),
            'gradient_shard_mb': rounded(grad_shard),
            'optimizer_state_shard_mb': rounded(opt_shard),
            'checkpoint_file': f'rank_{rank:02d}.distcp',
        })
    return shards


def run() -> None:
    rank_shards = build_rank_shards()
    full_training_state_mb = sum(TOTALS_MB.values())
    sharded_resume_peak_mb = sum(
        rank_shards[0][key]
        for key in ['parameter_shard_mb', 'gradient_shard_mb', 'optimizer_state_shard_mb']
    ) + 24.0
    no_offload_peak = 232.0
    cpu_offload_peak = 196.0
    metrics = {
        'backend': 'cpu-simulated-fsdp-checkpoint-offload',
        'deterministic_seed': 20260411,
        'rank_count': WORLD_SIZE,
        'state_dict_modes': {
            'full_state_dict': {
                'file_count': 1,
                'portable_export': True,
                'load_peak_mb': rounded(full_training_state_mb),
                'world_size_change': 'simple_single_process_merge_then_reshard',
                'best_for': 'portable inference export and single-process debugging',
            },
            'sharded_state_dict': {
                'file_count': WORLD_SIZE,
                'portable_export': False,
                'load_peak_mb': rounded(sharded_resume_peak_mb),
                'world_size_change': 'requires merge_or_reshard_aware_runtime',
                'best_for': 'same-cluster training resume under a tight memory budget',
            },
        },
        'best_resume_mode_by_peak': 'sharded_state_dict',
        'portable_export_mode': 'full_state_dict',
        'offload_policy': {
            'none': {
                'peak_gpu_memory_mb': no_offload_peak,
                'step_time_ms': 1280,
                'transfer_ms': 0,
            },
            'cpu_optimizer_offload': {
                'peak_gpu_memory_mb': cpu_offload_peak,
                'step_time_ms': 1510,
                'transfer_ms': 230,
            },
        },
        'rank_shards': rank_shards,
        'checkpoint_load_order': [
            'read_rank_local_shard',
            'materialize_param_shard',
            'attach_optimizer_shard',
            'optionally_merge_full_state_for_export',
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
