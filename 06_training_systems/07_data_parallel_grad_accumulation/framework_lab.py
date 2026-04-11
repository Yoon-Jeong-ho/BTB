from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "framework_metrics.json"

RANK_COUNT = 4
LOCAL_BATCH = 6
ACCUMULATION_STEPS = 3
MICROSTEPS = 6
TOKENS_PER_SAMPLE = 96
PARAMETER_MB = 384.0
GRADIENT_MB = 384.0
OPTIMIZER_MB = 768.0


def rounded(value: float) -> float:
    return round(value, 6)


def build_rank_windows() -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for rank in range(RANK_COUNT):
        windows.append(
            {
                "rank": rank,
                "data_shard": f"dataset_shard_{rank}",
                "local_batch": LOCAL_BATCH,
                "microsteps_before_sync": ACCUMULATION_STEPS,
                "gradient_buffer_mb": rounded(GRADIENT_MB),
                "accumulation_slots": [
                    {"slot": 1, "operation": "local_backward_no_sync"},
                    {"slot": 2, "operation": "local_backward_no_sync"},
                    {"slot": 3, "operation": "boundary_all_reduce_gradients"},
                ],
                "replica_contract": "full gradients are replicated per rank; data shards differ, gradient buffers are not sharded in DDP",
            }
        )
    return windows


def run() -> dict[str, object]:
    global_batch = LOCAL_BATCH * RANK_COUNT
    effective_batch = global_batch * ACCUMULATION_STEPS
    optimizer_steps = MICROSTEPS // ACCUMULATION_STEPS
    tokens_per_microstep = global_batch * TOKENS_PER_SAMPLE
    tokens_per_optimizer_step = effective_batch * TOKENS_PER_SAMPLE
    per_rank_parameter = PARAMETER_MB
    per_rank_gradient = GRADIENT_MB
    per_rank_optimizer = OPTIMIZER_MB

    metrics: dict[str, object] = {
        "status": "runnable",
        "framework": "deterministic_cpu_data_parallel_grad_accum_sim",
        "backend": "cpu_fallback",
        "gpu_probe": {
            "required": False,
            "behavior": "CPU fallback is the canonical path; GPU availability is intentionally not required for this unit.",
        },
        "rank_count": RANK_COUNT,
        "local_batch_size": LOCAL_BATCH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "microstep_count": MICROSTEPS,
        "global_batch_per_microstep": global_batch,
        "effective_batch_per_optimizer_step": effective_batch,
        "optimizer_step_cadence": {
            "microsteps_per_optimizer_step": ACCUMULATION_STEPS,
            "optimizer_steps": optimizer_steps,
            "scheduler_timing": "one scheduler step per optimizer step, not per microstep",
        },
        "collectives": [
            "local_backward_no_sync",
            "boundary_all_reduce_gradients",
            "optimizer_step",
        ],
        "communication_model": {
            "every_step_sync_calls": MICROSTEPS,
            "deferred_sync_calls": optimizer_steps,
            "estimated_gradient_payload_mb_per_sync": rounded(GRADIENT_MB),
            "estimated_payload_mb_every_step": rounded(MICROSTEPS * GRADIENT_MB),
            "estimated_payload_mb_deferred": rounded(optimizer_steps * GRADIENT_MB),
            "interpretation": "no_sync reduces communication cadence but does not remove the boundary all-reduce contract",
        },
        "memory_model": {
            "per_rank_parameter_replica_mb": rounded(per_rank_parameter),
            "per_rank_gradient_buffer_mb": rounded(per_rank_gradient),
            "per_rank_optimizer_state_mb": rounded(per_rank_optimizer),
            "local_batch_activation_proxy_mb": rounded(LOCAL_BATCH * 11.0),
            "same_effective_batch_large_local_activation_proxy_mb": rounded(LOCAL_BATCH * ACCUMULATION_STEPS * 11.0),
            "why_accumulation_fits": "activation peak follows local batch while optimizer sees the larger effective batch",
        },
        "throughput_model": {
            "tokens_per_microstep": tokens_per_microstep,
            "tokens_per_optimizer_step": tokens_per_optimizer_step,
            "forward_backward_passes_per_optimizer_step": ACCUMULATION_STEPS,
            "risk": "larger effective batch does not imply better wall-clock throughput because more microsteps feed one optimizer step",
        },
        "optimizer_dynamics": {
            "loss_normalization": "divide loss by accumulation_steps before backward",
            "gradient_clipping": "clip after boundary_all_reduce_gradients and before optimizer_step",
            "noise_scale": "closer to a larger batch regime, but runtime and scheduler traces remain different",
        },
        "scheduler_policy": "scheduler_steps_on_optimizer_step",
        "rank_windows": build_rank_windows(),
        "relations": {
            "tensor_parallelism": "tensor parallel splits layer math; this data parallel simulation keeps full model replicas and splits batch shards",
            "pipeline_parallelism": "pipeline microbatches flow across stages; grad accumulation microsteps delay optimizer cadence inside a replica group",
            "hybrid_parallelism": "data parallel axis still defines batch budget after tensor/pipeline/state-sharding axes are added",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
