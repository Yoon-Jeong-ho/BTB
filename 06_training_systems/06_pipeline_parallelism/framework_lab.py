from __future__ import annotations

import json
from pathlib import Path

from scratch_lab import FP32_BYTES, build_layers, partition_layers


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "framework_metrics.json"


DoneTimes = dict[tuple[int, int], int]


def _completed(done: DoneTimes, key: tuple[int, int], slot: int) -> bool:
    return key in done and done[key] <= slot


def build_1f1b_schedule(num_stages: int, microbatches: int) -> tuple[list[dict[str, object]], list[int]]:
    """Greedy dependency-valid 1F1B-like schedule.

    Each stage can run one unit operation per slot. A backward operation is chosen
    before a forward operation when both are dependency-ready, which captures the
    memory-saving instinct of 1F1B without requiring any distributed runtime.
    """
    forward_done: DoneTimes = {}
    backward_done: DoneTimes = {}
    trace: list[dict[str, object]] = []
    peak_saved_activations = [0 for _ in range(num_stages)]
    expected_ops = 2 * num_stages * microbatches

    while len(forward_done) + len(backward_done) < expected_ops:
        slot = len(trace)
        decisions: list[tuple[str, int | None]] = []
        for stage in range(num_stages):
            backward = _next_backward(stage, num_stages, microbatches, slot, forward_done, backward_done)
            if backward is not None:
                decisions.append(("B", backward))
                continue

            forward = _next_forward(stage, microbatches, slot, forward_done)
            if forward is not None:
                decisions.append(("F", forward))
            else:
                decisions.append(("idle", None))

        for stage, (kind, microbatch) in enumerate(decisions):
            if microbatch is None:
                continue
            if kind == "F":
                forward_done[(microbatch, stage)] = slot + 1
            elif kind == "B":
                backward_done[(microbatch, stage)] = slot + 1

        for stage in range(num_stages):
            saved = sum(
                1
                for microbatch in range(microbatches)
                if (microbatch, stage) in forward_done and (microbatch, stage) not in backward_done
            )
            peak_saved_activations[stage] = max(peak_saved_activations[stage], saved)

        trace.append(
            {
                "slot": slot,
                "stage_ops": [
                    "idle" if microbatch is None else f"{kind}{microbatch}"
                    for kind, microbatch in decisions
                ],
            }
        )

    return trace, peak_saved_activations


def _next_forward(stage: int, microbatches: int, slot: int, forward_done: DoneTimes) -> int | None:
    for microbatch in range(microbatches):
        key = (microbatch, stage)
        if key in forward_done:
            continue
        if stage == 0 or _completed(forward_done, (microbatch, stage - 1), slot):
            return microbatch
    return None


def _next_backward(
    stage: int,
    num_stages: int,
    microbatches: int,
    slot: int,
    forward_done: DoneTimes,
    backward_done: DoneTimes,
) -> int | None:
    for microbatch in range(microbatches):
        key = (microbatch, stage)
        if key in backward_done or not _completed(forward_done, key, slot):
            continue
        if stage == num_stages - 1 or _completed(backward_done, (microbatch, stage + 1), slot):
            return microbatch
    return None


def schedule_metrics(trace: list[dict[str, object]], num_stages: int, microbatches: int) -> dict[str, object]:
    total_slots = len(trace)
    active_slots = sum(
        1
        for slot in trace
        for op in slot["stage_ops"]
        if op != "idle"
    )
    total_stage_slots = total_slots * num_stages
    idle_slots = total_stage_slots - active_slots
    return {
        "total_time_slots": total_slots,
        "active_stage_slots": active_slots,
        "idle_stage_slots": idle_slots,
        "bubble_fraction": round(idle_slots / total_stage_slots, 4),
        "throughput_microbatches_per_slot": round(microbatches / total_slots, 4),
        "forward_ops": microbatches * num_stages,
        "backward_ops": microbatches * num_stages,
    }


def run() -> dict[str, object]:
    num_stages = 4
    microbatches = 8
    layers = build_layers()
    stages = partition_layers(layers, [3, 6, 9])
    trace, peak_saved = build_1f1b_schedule(num_stages, microbatches)
    metrics = schedule_metrics(trace, num_stages, microbatches)
    stage_compute_units = [int(stage["compute_units"]) for stage in stages]
    boundary_payloads = [
        int(stage["boundary_activation_elements"])
        for stage in stages[:-1]
    ]
    forward_messages = (num_stages - 1) * microbatches
    backward_messages = (num_stages - 1) * microbatches
    estimated_transfer_bytes = (sum(boundary_payloads) * microbatches * 2) * FP32_BYTES
    bottleneck = max(stage_compute_units)
    ideal_no_bubble_slots = 2 * microbatches

    report: dict[str, object] = {
        "status": "runnable",
        "framework": "deterministic_cpu_pipeline_parallel_sim",
        "schedule_policy": "1F1B_greedy_dependency_sim",
        "num_stages": num_stages,
        "microbatches": microbatches,
        "partition_plan": stages,
        "schedule_metrics": {
            **metrics,
            "ideal_no_bubble_slots_per_stage": ideal_no_bubble_slots,
            "bottleneck_stage": stage_compute_units.index(bottleneck),
            "bottleneck_stage_compute_units": bottleneck,
        },
        "schedule_trace_head": trace[:10],
        "schedule_trace_tail": trace[-6:],
        "transfers_per_boundary": [
            "forward_activation_send",
            "backward_gradient_recv",
        ],
        "activation_transfer_model": {
            "forward_messages": forward_messages,
            "backward_messages": backward_messages,
            "total_messages": forward_messages + backward_messages,
            "boundary_payload_elements": boundary_payloads,
            "estimated_bytes": estimated_transfer_bytes,
            "note": "simulated send/recv accounting only; no device communication is performed",
        },
        "activation_memory_model": {
            "gpipe_peak_saved_microbatches": microbatches,
            "one_f1b_peak_saved_microbatches": max(peak_saved),
            "peak_saved_by_stage": peak_saved,
            "interpretation": "1F1B starts backward earlier, so fewer microbatch activations remain live than GPipe all-forward-then-backward.",
        },
        "partitioning_concerns": {
            "stage_compute_units": stage_compute_units,
            "max_over_min_stage_compute": round(max(stage_compute_units) / min(stage_compute_units), 4),
            "bottleneck_warning": "pipeline throughput is bounded by the slowest stage plus boundary transfer cost",
            "boundary_warning": "skip/residual tensors that cross a boundary must preserve shape, dtype, and ordering",
        },
        "relations": {
            "tensor_parallelism": "splits inside a layer; this simulation splits layer ranges into pipeline stages",
            "data_parallelism": "replicates the pipeline replica over batch shards; optimizer cadence still depends on all microbatches",
            "hybrid_parallelism": "combines pipeline stages with tensor/data/state-sharding axes when one split axis is insufficient",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
