from __future__ import annotations

import json
from pathlib import Path

from scratch_lab import annotate_candidate, candidate_topologies, select_candidate


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "framework_metrics.json"


def score_candidate(candidate: dict[str, object]) -> dict[str, object]:
    memory = candidate["memory_budget"]
    communication = candidate["communication_budget"]
    margin = float(memory["memory_margin_gb"])
    link_pressure = float(communication["inter_node_link_pressure_score"])
    bubble = float(communication["bubble_fraction"])
    imbalance = float(memory["stage_compute_imbalance"])
    checkpoint_bonus = 0.06 if "reshard" in str(candidate["checkpoint_contract"]) else 0.02
    intra_node_bonus = 0.08 if communication["tensor_parallel_kept_intra_node"] else -0.25
    tp = int(candidate["tensor_parallel"])
    pp = int(candidate["pipeline_parallel"])
    dp = int(candidate["data_parallel"])
    balanced_axis_bonus = max(0.0, 0.16 - abs(tp - 4) * 0.035 - abs(pp - 2) * 0.045 - abs(dp - 8) * 0.015)
    memory_score = min(1.0, max(0.0, margin / 20.0))
    throughput_score = max(0.0, 1.0 - link_pressure / 80.0 - bubble - (imbalance - 1.0) * 0.55)
    total = round(memory_score * 0.25 + throughput_score * 0.45 + checkpoint_bonus + intra_node_bonus + balanced_axis_bonus, 4)
    return {
        "candidate": candidate["name"],
        "score": total,
        "memory_score": round(memory_score, 4),
        "throughput_score": round(throughput_score, 4),
        "checkpoint_portability_bonus": checkpoint_bonus,
        "intra_node_tp_bonus": intra_node_bonus,
        "dominant_bottleneck": communication["primary_risk"],
        "pedagogical_signal": (
            "best-balanced hybrid topology"
            if total > 0.70
            else "useful contrast case for bottleneck reasoning"
        ),
    }


def run() -> dict[str, object]:
    model = {
        "name": "decoder_only_llm_framework_case",
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
    annotated = [annotate_candidate(candidate, model, hardware) for candidate in candidate_topologies()]
    preferred = select_candidate(annotated)
    scores = [score_candidate(candidate) for candidate in annotated]
    scores = sorted(scores, key=lambda item: float(item["score"]), reverse=True)

    report: dict[str, object] = {
        "status": "runnable",
        "framework": "deterministic_cpu_hybrid_parallel_topology_sim",
        "world_size": hardware["total_gpus"],
        "device_mesh_axes": ["data_parallel", "tensor_parallel", "pipeline_parallel", "fsdp_state_sharding"],
        "preferred_candidate": preferred["name"],
        "rank_mesh_contract": {
            "rank_order": "dp_outer / pp_middle / tp_inner",
            "tp_inner_reason": "tensor-parallel all-reduce/all-gather is latency-sensitive, so keep it inside fast node-local links",
            "pp_middle_reason": "pipeline stages can cross node boundaries when activation payload and bubble are budgeted",
            "dp_fsdp_outer_reason": "data replica and FSDP shard groups define batch cadence, state residency, and checkpoint remap contract",
        },
        "candidate_scores": scores,
        "communication_tradeoffs": {
            "fast_link_axis": "tensor_parallel",
            "slow_link_candidates": ["pipeline_parallel", "data_parallel_gradient_sync"],
            "collectives_to_profile": [
                "tp_all_reduce",
                "fsdp_all_gather",
                "fsdp_reduce_scatter",
                "dp_gradient_all_reduce",
                "pipeline_send_recv",
            ],
        },
        "bottleneck_reasoning": {
            "memory_fit": "FSDP/state sharding, tensor split, and pipeline split all reduce different resident or peak memory terms.",
            "throughput": "TP traffic is frequent and should avoid slow links; PP bubble and DP sync cadence bound scaling after fit is achieved.",
            "checkpoint_portability": "The chosen topology must record DP/TP/PP/FSDP axes so checkpoint state can be reloaded or reshaped safely.",
        },
        "next_unit_profiling_hypotheses": [
            "If tp_all_reduce leaves the node, step latency rises before memory changes.",
            "If stage_compute_units are imbalanced, pipeline bubble remains visible even with enough microbatches.",
            "If FSDP all-gather overlaps poorly with pipeline boundary transfer, peak memory and link pressure spike together.",
        ],
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
