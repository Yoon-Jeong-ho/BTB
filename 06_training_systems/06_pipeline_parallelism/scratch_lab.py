from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "scratch_metrics.json"
SVG_PATH = ARTIFACT_DIR / "pipeline_schedule.svg"
FP32_BYTES = 4


Layer = dict[str, object]
Stage = dict[str, object]


def build_layers() -> list[Layer]:
    """Small deterministic transformer-like stack used for partition intuition."""
    names = ["embedding"] + [f"block_{index:02d}" for index in range(10)] + ["lm_head"]
    compute_units = [5, 6, 6, 7, 8, 8, 7, 6, 6, 5, 5, 7]
    activation_elements = [48, 64, 64, 80, 96, 96, 80, 72, 72, 64, 56, 40]
    return [
        {
            "name": name,
            "compute_units": compute,
            "activation_elements": activation,
        }
        for name, compute, activation in zip(names, compute_units, activation_elements)
    ]


def partition_layers(layers: list[Layer], boundaries: list[int]) -> list[Stage]:
    starts = [0, *boundaries]
    ends = [*boundaries, len(layers)]
    stages: list[Stage] = []
    for stage_id, (start, end) in enumerate(zip(starts, ends)):
        stage_layers = layers[start:end]
        boundary_activation = 0 if stage_id == len(starts) - 1 else int(stage_layers[-1]["activation_elements"])
        stages.append(
            {
                "stage": stage_id,
                "layer_range": [start, end - 1],
                "layers": [str(layer["name"]) for layer in stage_layers],
                "compute_units": sum(int(layer["compute_units"]) for layer in stage_layers),
                "parameter_units": len(stage_layers) * 100,
                "boundary_activation_elements": boundary_activation,
            }
        )
    return stages


def build_forward_schedule(num_stages: int, microbatches: int) -> list[dict[str, object]]:
    total_slots = microbatches + num_stages - 1
    trace: list[dict[str, object]] = []
    for slot in range(total_slots):
        stage_ops: list[str] = []
        for stage in range(num_stages):
            microbatch = slot - stage
            if 0 <= microbatch < microbatches:
                stage_ops.append(f"F{microbatch}")
            else:
                stage_ops.append("idle")
        trace.append({"slot": slot, "stage_ops": stage_ops})
    return trace


def stage_idle_counts(schedule: list[dict[str, object]], num_stages: int) -> list[int]:
    return [
        sum(1 for slot in schedule if slot["stage_ops"][stage] == "idle")
        for stage in range(num_stages)
    ]


def compute_schedule_summary(schedule: list[dict[str, object]], num_stages: int, microbatches: int) -> dict[str, object]:
    total_time_slots = len(schedule)
    active_stage_slots = microbatches * num_stages
    total_stage_slots = total_time_slots * num_stages
    idle_stage_slots = total_stage_slots - active_stage_slots
    steady_state_slots = max(0, microbatches - num_stages + 1)
    return {
        "policy": "forward_pipeline_fill_drain",
        "warmup_slots": num_stages - 1,
        "steady_state_slots": steady_state_slots,
        "cooldown_slots": num_stages - 1,
        "total_time_slots": total_time_slots,
        "active_stage_slots": active_stage_slots,
        "idle_stage_slots": idle_stage_slots,
        "bubble_fraction": round(idle_stage_slots / total_stage_slots, 4),
        "throughput_microbatches_per_slot": round(microbatches / total_time_slots, 4),
    }


def write_svg(schedule: list[dict[str, object]], stage_compute: list[int]) -> None:
    cell_w = 72
    cell_h = 34
    left = 118
    top = 78
    width = left + cell_w * len(schedule) + 32
    height = top + cell_h * len(stage_compute) + 82
    colors = {
        "active": "#bfdbfe",
        "idle": "#f3f4f6",
        "stroke": "#1f2937",
    }
    stroke = colors["stroke"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff7ed"/>',
        '<text x="24" y="34" font-family="monospace" font-size="18" fill="#7c2d12">Pipeline schedule: fill / steady / drain</text>',
        '<text x="24" y="56" font-family="monospace" font-size="12" fill="#374151">CPU-only deterministic simulation; each F# is one microbatch forward on one stage.</text>',
    ]
    for slot in schedule:
        x = left + int(slot["slot"]) * cell_w
        parts.append(
            f'<text x="{x + 20}" y="{top - 12}" font-family="monospace" font-size="11" fill="#374151">t{slot["slot"]}</text>'
        )
    for stage, compute in enumerate(stage_compute):
        y = top + stage * cell_h
        parts.append(
            f'<text x="24" y="{y + 22}" font-family="monospace" font-size="12" fill="#111827">stage {stage} ({compute}u)</text>'
        )
        for slot in schedule:
            op = str(slot["stage_ops"][stage])
            x = left + int(slot["slot"]) * cell_w
            fill = colors["idle"] if op == "idle" else colors["active"]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 6}" height="{cell_h - 6}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{x + 18}" y="{y + 20}" font-family="monospace" font-size="12" fill="#111827">{op}</text>'
            )
    legend_y = top + cell_h * len(stage_compute) + 34
    parts.extend(
        [
            f'<rect x="24" y="{legend_y - 14}" width="18" height="18" fill="#bfdbfe" stroke="#1f2937"/>',
            f'<text x="50" y="{legend_y}" font-family="monospace" font-size="12">active microbatch work</text>',
            f'<rect x="250" y="{legend_y - 14}" width="18" height="18" fill="#f3f4f6" stroke="#1f2937"/>',
            f'<text x="276" y="{legend_y}" font-family="monospace" font-size="12">pipeline bubble / idle slot</text>',
            "</svg>",
        ]
    )
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def run() -> dict[str, object]:
    microbatches = 6
    partition_boundaries = [4, 8]
    layers = build_layers()
    stages = partition_layers(layers, partition_boundaries)
    num_stages = len(stages)
    schedule = build_forward_schedule(num_stages, microbatches)
    summary = compute_schedule_summary(schedule, num_stages, microbatches)
    idle_counts = stage_idle_counts(schedule, num_stages)
    stage_compute = [int(stage["compute_units"]) for stage in stages]
    boundary_payloads = [
        int(stage["boundary_activation_elements"])
        for stage in stages[:-1]
    ]
    transfer_messages = (num_stages - 1) * microbatches
    estimated_transfer_bytes = sum(boundary_payloads) * microbatches * FP32_BYTES
    bottleneck_compute = max(stage_compute)
    total_pipeline_work = summary["total_time_slots"] * bottleneck_compute
    single_stage_work = microbatches * sum(stage_compute)

    metrics: dict[str, object] = {
        "status": "runnable",
        "simulation": "deterministic_cpu_pipeline_schedule",
        "num_layers": len(layers),
        "num_stages": num_stages,
        "microbatches": microbatches,
        "partition_plan": stages,
        "schedule_summary": {
            **summary,
            "bottleneck_stage": stage_compute.index(bottleneck_compute),
            "bottleneck_stage_compute_units": bottleneck_compute,
            "estimated_speedup_vs_single_stage_serial": round(single_stage_work / total_pipeline_work, 4),
        },
        "schedule_trace": schedule,
        "stage_observations": [
            {
                "stage": stage["stage"],
                "idle_slots": idle_counts[int(stage["stage"])],
                "compute_units": stage["compute_units"],
                "dominant_concern": (
                    "bottleneck compute" if int(stage["compute_units"]) == bottleneck_compute else "waits during fill/drain"
                ),
            }
            for stage in stages
        ],
        "activation_transfer": {
            "boundary_count": num_stages - 1,
            "messages_per_boundary": microbatches,
            "total_messages": transfer_messages,
            "boundary_payload_elements": boundary_payloads,
            "estimated_bytes": estimated_transfer_bytes,
            "contract": "send forward activations from stage i to i+1 for every microbatch",
            "note": "CPU-only byte estimate; no device or network runtime is used.",
        },
        "partition_balance": {
            "stage_compute_units": stage_compute,
            "max_over_min_stage_compute": round(max(stage_compute) / min(stage_compute), 4),
            "risk": "layer-count balance is not the same as compute/communication balance",
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
            "svg": str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_svg(schedule, stage_compute)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
