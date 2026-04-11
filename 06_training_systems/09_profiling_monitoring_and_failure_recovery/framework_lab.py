from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / 'artifacts' / 'framework-manual'
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'


INCIDENTS = [
    {
        'incident_id': 'inc_oom_eval_boundary',
        'symptom': 'CUDA out of memory during eval/save boundary',
        'classification': 'oom_memory_spike',
        'first_checks': [
            'peak_allocated_vs_reserved_memory',
            'phase_that_created_peak',
            'whether_reserved_memory_recovers_after_eval',
        ],
        'recoverable': True,
        'recommended_action': 'reduce_eval_microbatch_and_retry_from_current_checkpoint',
        'priority': 2,
    },
    {
        'incident_id': 'inc_rank2_stall_step126',
        'symptom': 'all ranks wait at collective while rank 2 heartbeat is stale',
        'classification': 'hang_or_straggler',
        'first_checks': [
            'per_rank_heartbeat',
            'last_collective_name',
            'rank_2_dataloader_and_host_logs',
            'checkpoint_writer_progress',
        ],
        'recoverable': True,
        'recommended_action': 'retry_from_last_good_checkpoint',
        'priority': 1,
    },
    {
        'incident_id': 'inc_loss_nan_after_resume',
        'symptom': 'loss becomes NaN in the first validation after resume',
        'classification': 'divergence_after_resume',
        'first_checks': [
            'optimizer_step_continuity',
            'lr_scheduler_state_continuity',
            'grad_norm_before_failure',
            'all_ranks_loaded_same_manifest',
        ],
        'recoverable': False,
        'recommended_action': 'quarantine_checkpoint_and_resume_from_previous_verified_manifest',
        'priority': 3,
    },
]


def choose_incident(incidents: list[dict[str, object]]) -> dict[str, object]:
    return sorted(incidents, key=lambda item: int(item['priority']))[0]


def build_report() -> dict[str, object]:
    selected = choose_incident(INCIDENTS)
    required_state = [
        'model_state',
        'optimizer_state',
        'scheduler_state',
        'scaler_state',
        'global_step',
        'sampler_state',
        'rng_state',
        'world_size_and_topology',
    ]
    validation_checks = {
        'manifest_hash_matches_all_ranks': True,
        'global_step_matches_checkpoint': True,
        'optimizer_step_is_continuous': True,
        'sampler_position_restored': True,
        'first_5_steps_have_finite_loss': True,
    }
    return {
        'status': 'runnable',
        'framework': 'cpu_deterministic_monitoring_recovery_sim',
        'deterministic_seed': 20260412,
        'monitoring_contract': {
            'required_signals': [
                'throughput_tokens_per_sec',
                'step_time_p50_p95',
                'loss_and_grad_norm',
                'gpu_memory_allocated_reserved',
                'per_rank_heartbeat',
                'checkpoint_freshness_minutes',
                'retry_count_and_failure_class',
            ],
            'alert_thresholds': {
                'throughput_drop_pct': 20,
                'step_time_p95_over_p50_ratio': 1.30,
                'heartbeat_lag_ms': 120,
                'checkpoint_freshness_minutes': 30,
                'grad_norm_spike_ratio': 2.50,
            },
            'sample_snapshot': {
                'throughput_tokens_per_sec': 35182.403433,
                'step_time_p50_ms': 102.0,
                'step_time_p95_ms': 135.0,
                'loss': 2.31,
                'grad_norm': 1.87,
                'gpu_memory_allocated_reserved': {'allocated_mb': 11080, 'reserved_mb': 13056},
                'per_rank_heartbeat': {'rank_2_lag_ms': 280},
                'checkpoint_freshness_minutes': 42,
            },
        },
        'failure_triage': {
            'taxonomy': ['oom_memory_spike', 'hang_or_straggler', 'divergence_after_resume', 'storage_checkpoint_failure'],
            'incidents': INCIDENTS,
            'selected_incident': selected,
            'decision_reason': 'heartbeat lag and communication wait moved before loss/grad signals, so classify as liveness/straggler first',
        },
        'checkpoint_manifest': {
            'checkpoint_id': 'ckpt_step_000120_verified',
            'format': 'sharded_state_plus_manifest',
            'required_state': required_state,
            'rank_files': [f'rank_{rank:02d}.distcp' for rank in range(4)],
            'atomic_write_contract': 'write_tmp_then_rename_manifest_last',
            'last_verified_global_step': 120,
        },
        'retry_policy': {
            'max_attempts': 3,
            'attempts_used': 1,
            'backoff': 'linear_60s_for_transient_liveness_failure',
            'do_not_retry_classes': ['corrupt_checkpoint_manifest', 'repeated_nan_after_verified_resume'],
        },
        'recovery_decision': {
            'action': selected['recommended_action'],
            'resume_from': 'ckpt_step_000120_verified',
            'skip_current_partial_checkpoint': True,
            'post_resume_validation': {
                'passed': all(validation_checks.values()),
                'checks': validation_checks,
                'validation_window_steps': 5,
            },
            'operator_note': 'Retry is safe because the manifest is verified and the selected failure is liveness, not numerical corruption.',
        },
    }


def run() -> dict[str, object]:
    report = build_report()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
