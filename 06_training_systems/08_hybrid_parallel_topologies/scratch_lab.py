from __future__ import annotations

import json
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "scratch_metrics.json"
SVG_PATH = ARTIFACT_DIR / "hybrid_topology_mesh.svg"
FP16_BYTES = 2
STATE_BYTES_PER_PARAM = 16  # fp16 param + grad + fp32 optimizer moments/master weight, simplified


TopologyCandidate = dict[str, object]


def candidate_topologies() -> list[TopologyCandidate]:
    """Deterministic topology candidates for a 64-GPU large-model training cluster."""
    return [
        {
            "name": "tp4_pp2_dp8_fsdp_hybrid",
            "data_parallel": 8,
            "tensor_parallel": 4,
            "pipeline_parallel": 2,
            "fsdp_mode": "hybrid_shard",
            "microbatches": 8,
            "grad_accum_steps": 4,
            "stage_compute_units": [58, 62],
            "boundary_activation_mb": [640],
            "checkpoint_contract": "reshard_by_dp_group_then_export_pipeline_stage",
        },
        {
            "name": "tp2_pp4_dp8_fsdp_full_shard",
            "data_parallel": 8,
            "tensor_parallel": 2,
            "pipeline_parallel": 4,
            "fsdp_mode": "full_shard",
            "microbatches": 8,
            "grad_accum_steps": 4,
            "stage_compute_units": [29, 31, 34, 26],
            "boundary_activation_mb": [360, 420, 300],
            "checkpoint_contract": "stage_local_state_dict_with_global_reshard_metadata",
        },
        {
            "name": "tp8_pp2_dp4_fsdp_hybrid",
            "data_parallel": 4,
            "tensor_parallel": 8,
            "pipeline_parallel": 2,
            "fsdp_mode": "hybrid_shard",
            "microbatches": 8,
            "grad_accum_steps": 8,
            "stage_compute_units": [60, 61],
            "boundary_activation_mb": [680],
            "checkpoint_contract": "tp_wide_stage_checkpoint_with_dp_outer_replica",
        },
    ]


def topology_world_size(candidate: TopologyCandidate) -> int:
    return int(candidate["data_parallel"]) * int(candidate["tensor_parallel"]) * int(candidate["pipeline_parallel"])


def tensor_parallel_crosses_node(candidate: TopologyCandidate, gpus_per_node: int) -> bool:
    return int(candidate["tensor_parallel"]) > gpus_per_node or gpus_per_node % int(candidate["tensor_parallel"]) != 0


def fsdp_shard_factor(candidate: TopologyCandidate) -> int:
    mode = str(candidate["fsdp_mode"])
    if mode == "full_shard":
        return int(candidate["data_parallel"])
    if mode == "hybrid_shard":
        # Small, local shard groups are easier to keep topology-aware in this CPU model.
        return min(4, int(candidate["data_parallel"]))
    return 1


def estimate_memory(candidate: TopologyCandidate, params_b: int, activation_base_gb: float) -> dict[str, object]:
    params = params_b * 1_000_000_000
    tp = int(candidate["tensor_parallel"])
    pp = int(candidate["pipeline_parallel"])
    shard = fsdp_shard_factor(candidate)
    stage_compute = [int(value) for value in candidate["stage_compute_units"]]
    imbalance = max(stage_compute) / min(stage_compute)

    dense_state_gb = params * STATE_BYTES_PER_PARAM / 1_000_000_000
    per_rank_state_gb = dense_state_gb / (tp * pp * shard)
    activation_peak_gb = activation_base_gb / pp * (1 + (imbalance - 1) * 0.35)
    fsdp_gather_peak_gb = (params * FP16_BYTES / 1_000_000_000) / (tp * pp * max(1, shard / 2))
    total_peak_gb = per_rank_state_gb + activation_peak_gb + fsdp_gather_peak_gb

    return {
        "dense_state_gb": round(dense_state_gb, 2),
        "fsdp_shard_factor": shard,
        "per_rank_state_gb": round(per_rank_state_gb, 2),
        "activation_peak_gb": round(activation_peak_gb, 2),
        "fsdp_all_gather_peak_gb": round(fsdp_gather_peak_gb, 2),
        "estimated_per_rank_peak_gb": round(total_peak_gb, 2),
        "stage_compute_imbalance": round(imbalance, 4),
    }


def estimate_communication(candidate: TopologyCandidate, gpus_per_node: int, inter_node_gbps: int) -> dict[str, object]:
    tp = int(candidate["tensor_parallel"])
    pp = int(candidate["pipeline_parallel"])
    dp = int(candidate["data_parallel"])
    microbatches = int(candidate["microbatches"])
    boundary_activation_mb = [int(value) for value in candidate["boundary_activation_mb"]]
    tp_cross_node = tensor_parallel_crosses_node(candidate, gpus_per_node)
    stage_compute = [int(value) for value in candidate["stage_compute_units"]]

    tp_collective_mb = 192 * tp
    fsdp_collective_mb = 840 * fsdp_shard_factor(candidate)
    dp_gradient_sync_mb = 512 * dp
    pipeline_send_recv_mb = sum(boundary_activation_mb) * microbatches
    inter_node_pipeline_fraction = 0.65 if pp > 2 else 0.35
    cross_node_bytes_mb = pipeline_send_recv_mb * inter_node_pipeline_fraction
    if tp_cross_node:
        cross_node_bytes_mb += tp_collective_mb

    link_pressure = round(cross_node_bytes_mb / inter_node_gbps, 4)
    bubble_fraction = round((pp - 1) / (microbatches + pp - 1), 4)
    imbalance_penalty = round(max(stage_compute) / min(stage_compute) - 1, 4)

    hotspots = [
        "tp_all_reduce" if not tp_cross_node else "tp_all_reduce_cross_node_risk",
        "fsdp_all_gather_reduce_scatter",
        "pipeline_activation_send_recv",
        "dp_gradient_sync",
    ]
    primary = "tensor_parallel_crosses_slow_link" if tp_cross_node else "pipeline_or_fsdp_overlap"
    if bubble_fraction > 0.25:
        primary = "pipeline_bubble_and_stage_imbalance"

    return {
        "tensor_parallel_kept_intra_node": not tp_cross_node,
        "tp_collective_mb_per_step": tp_collective_mb,
        "fsdp_collective_mb_per_step": fsdp_collective_mb,
        "dp_gradient_sync_mb_per_step": dp_gradient_sync_mb,
        "pipeline_send_recv_mb_per_step": pipeline_send_recv_mb,
        "estimated_cross_node_mb_per_step": round(cross_node_bytes_mb, 2),
        "inter_node_link_pressure_score": link_pressure,
        "bubble_fraction": bubble_fraction,
        "stage_imbalance_penalty": imbalance_penalty,
        "communication_hotspots": hotspots,
        "primary_risk": primary,
    }


def annotate_candidate(candidate: TopologyCandidate, model: dict[str, object], hardware: dict[str, object]) -> dict[str, object]:
    memory = estimate_memory(candidate, int(model["params_b"]), float(model["activation_base_gb"]))
    communication = estimate_communication(candidate, int(hardware["gpus_per_node"]), int(hardware["inter_node_link_gbps"]))
    peak = float(memory["estimated_per_rank_peak_gb"])
    memory_margin = round(float(hardware["memory_per_gpu_gb"]) - peak, 2)
    topology_fit = topology_world_size(candidate) == int(hardware["total_gpus"])
    bottleneck_reasoning = (
        "TP collectives stay on the fast intra-node link; remaining risk is overlap among FSDP gather, "
        "pipeline activation transfer, and outer data-parallel sync."
        if communication["tensor_parallel_kept_intra_node"]
        else "Tensor-parallel traffic crosses a slower inter-node boundary, so latency-sensitive layer collectives dominate."
    )
    if float(communication["bubble_fraction"]) > 0.25:
        bottleneck_reasoning = "Deeper pipeline split lowers stage memory but raises bubble and load-balance sensitivity."

    return {
        **candidate,
        "world_size": topology_world_size(candidate),
        "topology_fit": topology_fit,
        "axis_product": f"DP{candidate['data_parallel']} x TP{candidate['tensor_parallel']} x PP{candidate['pipeline_parallel']}",
        "memory_budget": {
            **memory,
            "memory_per_gpu_gb": hardware["memory_per_gpu_gb"],
            "memory_margin_gb": memory_margin,
            "fits_memory_budget": memory_margin > 0,
        },
        "communication_budget": communication,
        "bottleneck_reasoning": bottleneck_reasoning,
    }


def select_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    def score(candidate: dict[str, object]) -> tuple[bool, float, float, float, float]:
        # Once a candidate fits in memory, prefer the balanced teaching topology:
        # TP4 remains node-local on 8-GPU nodes, PP2 limits bubble, and DP8 keeps
        # the batch/state-sharding axis visible. Extra memory margin alone should
        # not beat a topology with much higher communication and scheduling risk.
        memory_margin = float(candidate["memory_budget"]["memory_margin_gb"])
        link_pressure = float(candidate["communication_budget"]["inter_node_link_pressure_score"])
        bubble = float(candidate["communication_budget"]["bubble_fraction"])
        tp = int(candidate["tensor_parallel"])
        pp = int(candidate["pipeline_parallel"])
        dp = int(candidate["data_parallel"])
        balanced_axis_bonus = 20.0 - abs(tp - 4) * 3.0 - abs(pp - 2) * 4.0 - abs(dp - 8)
        capped_margin = min(memory_margin, 20.0)
        return (memory_margin > 0, balanced_axis_bonus, -link_pressure, -bubble, capped_margin)

    return max(candidates, key=score)


def write_svg(selected_name: str) -> None:
    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"860\" height=\"360\" viewBox=\"0 0 860 360\">
  <rect width=\"860\" height=\"360\" fill=\"#f8fafc\"/>
  <text x=\"24\" y=\"36\" font-family=\"monospace\" font-size=\"18\" fill=\"#0f172a\">Hybrid parallel topology mesh (CPU planning artifact)</text>
  <text x=\"24\" y=\"62\" font-family=\"monospace\" font-size=\"12\" fill=\"#475569\">Selected: {selected_name}; TP stays inside node, PP crosses stage boundaries, DP/FSDP form outer groups.</text>
  <rect x=\"44\" y=\"96\" width=\"350\" height=\"190\" rx=\"12\" fill=\"#e0f2fe\" stroke=\"#0369a1\"/>
  <rect x=\"466\" y=\"96\" width=\"350\" height=\"190\" rx=\"12\" fill=\"#ecfccb\" stroke=\"#4d7c0f\"/>
  <text x=\"62\" y=\"124\" font-family=\"monospace\" font-size=\"14\" fill=\"#0c4a6e\">Pipeline stage 0</text>
  <text x=\"484\" y=\"124\" font-family=\"monospace\" font-size=\"14\" fill=\"#365314\">Pipeline stage 1</text>
  <text x=\"156\" y=\"252\" font-family=\"monospace\" font-size=\"12\" fill=\"#075985\">TP4 group: all-reduce on fast intra-node links</text>
  <text x=\"558\" y=\"252\" font-family=\"monospace\" font-size=\"12\" fill=\"#3f6212\">TP4 group: mirrored for next stage</text>
  <line x1=\"394\" y1=\"190\" x2=\"466\" y2=\"190\" stroke=\"#be123c\" stroke-width=\"4\" marker-end=\"url(#arrow)\"/>
  <text x=\"398\" y=\"176\" font-family=\"monospace\" font-size=\"11\" fill=\"#be123c\">PP send/recv</text>
  <path d=\"M92 306 C230 334 626 334 764 306\" fill=\"none\" stroke=\"#7c3aed\" stroke-width=\"3\" stroke-dasharray=\"8 6\"/>
  <text x=\"280\" y=\"338\" font-family=\"monospace\" font-size=\"12\" fill=\"#5b21b6\">DP/FSDP outer groups: shard state, sync gradients, preserve checkpoint contract</text>
  <defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 z\" fill=\"#be123c\"/></marker></defs>
  <g font-family=\"monospace\" font-size=\"12\" fill=\"#111827\">
    <text x=\"84\" y=\"162\">rank 0</text><text x=\"174\" y=\"162\">rank 1</text><text x=\"264\" y=\"162\">rank 2</text><text x=\"334\" y=\"162\">rank 3</text>
    <text x=\"506\" y=\"162\">rank 4</text><text x=\"596\" y=\"162\">rank 5</text><text x=\"686\" y=\"162\">rank 6</text><text x=\"756\" y=\"162\">rank 7</text>
    <text x=\"74\" y=\"214\">node-local fast collective domain</text><text x=\"500\" y=\"214\">node-local fast collective domain</text>
  </g>
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")


def run() -> dict[str, object]:
    model = {
        "name": "decoder_only_llm_planning_case",
        "params_b": 70,
        "sequence_length": 8192,
        "target_global_batch": 1024,
        "activation_base_gb": 44.0,
    }
    hardware = {
        "nodes": 8,
        "gpus_per_node": 8,
        "total_gpus": 64,
        "memory_per_gpu_gb": 80,
        "intra_node_link": "NVLink/NVSwitch",
        "inter_node_link": "InfiniBand",
        "inter_node_link_gbps": 100,
    }
    candidates = [annotate_candidate(candidate, model, hardware) for candidate in candidate_topologies()]
    selected = select_candidate(candidates)

    metrics: dict[str, object] = {
        "status": "runnable",
        "simulation": "deterministic_cpu_hybrid_topology_planner",
        "cpu_safe_simulation": True,
        "model": model,
        "hardware": hardware,
        "parallel_axes": {
            "data_parallel": "replica / batch axis and gradient synchronization cadence",
            "tensor_parallel": "intra-layer matrix and attention-head split; latency-sensitive collectives",
            "pipeline_parallel": "layer-stage split plus microbatch time-axis schedule",
            "fsdp_state_sharding": "parameter/gradient/optimizer state residency and checkpoint lifecycle",
        },
        "candidate_topologies": candidates,
        "preferred_candidate": selected["name"],
        "selection_summary": {
            "axis_product": selected["axis_product"],
            "memory_margin_gb": selected["memory_budget"]["memory_margin_gb"],
            "primary_risk": selected["communication_budget"]["primary_risk"],
            "reason": [
                "keeps tensor-parallel collectives inside fast node-local links",
                "uses pipeline depth 2 to reduce model residency without excessive bubble",
                "keeps FSDP/state sharding as an explicit checkpoint-aware memory axis",
            ],
            "profiling_focus_next_unit": [
                "tp_all_reduce latency and overlap",
                "pipeline stage idle/bubble time",
                "fsdp all-gather peak memory",
                "checkpoint save/load rank remapping",
            ],
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
            "svg": str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_svg(str(selected["name"]))
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
