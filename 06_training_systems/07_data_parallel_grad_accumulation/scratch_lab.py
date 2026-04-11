from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "scratch_metrics.json"
SVG_PATH = ARTIFACT_DIR / "data_parallel_grad_accumulation.svg"

WORLD_SIZE = 4
LOCAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
MICROSTEP_COUNT = 8
TOKENS_PER_SAMPLE = 128
BASE_MODEL_MEMORY_MB = 320.0
ACTIVATION_MB_PER_SAMPLE = 9.5
GRADIENT_BUFFER_MB = 96.0
OPTIMIZER_STATE_MB = 192.0


def rounded(value: float) -> float:
    return round(value, 6)


def build_accumulation_trace() -> list[dict[str, object]]:
    trace: list[dict[str, object]] = []
    for index in range(MICROSTEP_COUNT):
        microstep = index + 1
        accumulation_slot = index % GRAD_ACCUM_STEPS + 1
        sync_gradients = accumulation_slot == GRAD_ACCUM_STEPS
        optimizer_step = sync_gradients
        trace.append(
            {
                "microstep": microstep,
                "accumulation_slot": accumulation_slot,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_batch_this_microstep": LOCAL_BATCH_SIZE * WORLD_SIZE,
                "loss_scale": rounded(1 / GRAD_ACCUM_STEPS),
                "sync_gradients": sync_gradients,
                "collective": "all_reduce_gradients" if sync_gradients else "no_sync_deferred",
                "optimizer_step": optimizer_step,
                "scheduler_step": optimizer_step,
            }
        )
    return trace


def write_svg(trace: list[dict[str, object]]) -> None:
    cell_w = 92
    cell_h = 42
    left = 44
    top = 82
    width = left + cell_w * GRAD_ACCUM_STEPS + 80
    height = top + cell_h * 2 + 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="34" font-family="monospace" font-size="17" fill="#0f172a">Data parallel + grad accumulation cadence</text>',
        '<text x="24" y="56" font-family="monospace" font-size="12" fill="#334155">no_sync for microsteps 1-3, all-reduce + optimizer step at the boundary</text>',
    ]
    for row in range(2):
        for col in range(GRAD_ACCUM_STEPS):
            item = trace[row * GRAD_ACCUM_STEPS + col]
            x = left + col * cell_w
            y = top + row * cell_h
            boundary = bool(item["sync_gradients"])
            fill = "#bbf7d0" if boundary else "#dbeafe"
            label = "sync+step" if boundary else "no_sync"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="5" fill="{fill}" stroke="#1e293b"/>')
            parts.append(f'<text x="{x + 10}" y="{y + 15}" font-family="monospace" font-size="11">m{item["microstep"]}</text>')
            parts.append(f'<text x="{x + 10}" y="{y + 29}" font-family="monospace" font-size="11">{label}</text>')
    parts.extend(
        [
            f'<text x="24" y="{height - 48}" font-family="monospace" font-size="12" fill="#334155">effective batch = local batch {LOCAL_BATCH_SIZE} × world size {WORLD_SIZE} × accum {GRAD_ACCUM_STEPS} = {LOCAL_BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM_STEPS}</text>',
            f'<text x="24" y="{height - 28}" font-family="monospace" font-size="12" fill="#334155">loss normalization scale per microstep = 1/{GRAD_ACCUM_STEPS}</text>',
            '</svg>',
        ]
    )
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def run() -> dict[str, object]:
    global_batch = LOCAL_BATCH_SIZE * WORLD_SIZE
    effective_batch = global_batch * GRAD_ACCUM_STEPS
    optimizer_steps = MICROSTEP_COUNT // GRAD_ACCUM_STEPS
    trace = build_accumulation_trace()

    small_activation_peak = LOCAL_BATCH_SIZE * ACTIVATION_MB_PER_SAMPLE
    equivalent_large_local_batch = LOCAL_BATCH_SIZE * GRAD_ACCUM_STEPS
    large_activation_peak = equivalent_large_local_batch * ACTIVATION_MB_PER_SAMPLE

    metrics: dict[str, object] = {
        "status": "runnable",
        "cpu_safe_simulation": True,
        "simulation": "deterministic_data_parallel_grad_accum_cadence",
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "global_batch_per_microstep": global_batch,
        "effective_batch_per_optimizer_step": effective_batch,
        "microstep_count": MICROSTEP_COUNT,
        "optimizer_step_count": optimizer_steps,
        "tokens_per_sample": TOKENS_PER_SAMPLE,
        "tokens_per_microstep_global": global_batch * TOKENS_PER_SAMPLE,
        "tokens_per_optimizer_step": effective_batch * TOKENS_PER_SAMPLE,
        "accumulation_trace": trace,
        "sync_policy_comparison": {
            "every_step_all_reduce_count": MICROSTEP_COUNT,
            "deferred_sync_all_reduce_count": optimizer_steps,
            "saved_all_reduce_calls": MICROSTEP_COUNT - optimizer_steps,
            "policy": "deferred sync / no_sync until accumulation boundary",
        },
        "loss_normalization": {
            "scale_per_microstep": rounded(1 / GRAD_ACCUM_STEPS),
            "reason": "divide the microstep loss so accumulated gradients match one large effective batch scale",
        },
        "gradient_clipping": {
            "recommended_timing": "clip_after_accumulation_boundary",
            "reason": "clip the aggregate gradient once, after deferred all-reduce and before optimizer_step",
        },
        "memory_model_mb": {
            "base_model_replica": BASE_MODEL_MEMORY_MB,
            "gradient_buffer": GRADIENT_BUFFER_MB,
            "optimizer_state": OPTIMIZER_STATE_MB,
            "small_local_batch_activation_peak": rounded(small_activation_peak),
            "equivalent_large_local_batch_activation_peak": rounded(large_activation_peak),
            "small_local_batch_with_accumulation_peak": rounded(BASE_MODEL_MEMORY_MB + GRADIENT_BUFFER_MB + OPTIMIZER_STATE_MB + small_activation_peak),
            "equivalent_large_local_batch_peak": rounded(BASE_MODEL_MEMORY_MB + GRADIENT_BUFFER_MB + OPTIMIZER_STATE_MB + large_activation_peak),
            "interpretation": "accumulation keeps activation memory tied to local microbatch size, not effective batch size",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
            "svg": str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_svg(trace)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
