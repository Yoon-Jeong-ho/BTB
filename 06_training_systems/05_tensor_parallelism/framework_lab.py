from __future__ import annotations

import json
from pathlib import Path

from scratch_lab import (
    FP32_BYTES,
    concat_columns,
    make_matrix,
    matmul,
    max_abs_diff,
    shape,
    split_columns,
    split_rows,
    sum_matrices,
)


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "framework_metrics.json"


def run() -> dict[str, object]:
    tp_world_size = 4
    sequence_length = 3
    hidden_size = 8
    intermediate_size = 16
    num_attention_heads = 8
    heads_per_rank = num_attention_heads // tp_world_size

    block_input = make_matrix(sequence_length, hidden_size, offset=2)
    up_projection = make_matrix(hidden_size, intermediate_size, offset=7)
    down_projection = make_matrix(intermediate_size, hidden_size, offset=13)

    up_shards = split_columns(up_projection, tp_world_size)
    hidden_shards = [matmul(block_input, shard) for shard in up_shards]
    gathered_hidden = concat_columns(hidden_shards)

    down_input_shards = split_columns(gathered_hidden, tp_world_size)
    down_weight_shards = split_rows(down_projection, tp_world_size)
    partial_outputs = [
        matmul(input_shard, weight_shard)
        for input_shard, weight_shard in zip(down_input_shards, down_weight_shards)
    ]
    tp_output = sum_matrices(partial_outputs)
    dense_output = matmul(matmul(block_input, up_projection), down_projection)

    dense_parameter_elements = hidden_size * intermediate_size + intermediate_size * hidden_size
    per_rank_parameter_elements = dense_parameter_elements // tp_world_size
    activation_elements_per_rank = sequence_length * (intermediate_size // tp_world_size)
    all_gather_elements = sequence_length * intermediate_size
    all_reduce_elements = sum(shape(partial)[0] * shape(partial)[1] for partial in partial_outputs)
    communication_bytes = (all_gather_elements + all_reduce_elements) * FP32_BYTES
    compute_units = sequence_length * dense_parameter_elements
    communication_share = round(communication_bytes / (communication_bytes + compute_units), 6)

    metrics: dict[str, object] = {
        "status": "runnable",
        "framework": "deterministic_cpu_tensor_parallel_sim",
        "tp_world_size": tp_world_size,
        "attention_partition": {
            "num_heads_total": num_attention_heads,
            "heads_per_rank": heads_per_rank,
            "head_dim": hidden_size // num_attention_heads,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
        },
        "matrix_shards": {
            "mlp_up_column_parallel_weight_per_rank": shape(up_shards[0]),
            "mlp_up_activation_per_rank": shape(hidden_shards[0]),
            "mlp_down_row_parallel_weight_per_rank": shape(down_weight_shards[0]),
            "mlp_down_input_activation_per_rank": shape(down_input_shards[0]),
            "mlp_down_partial_output_per_rank": shape(partial_outputs[0]),
        },
        "collectives_per_block": [
            "all_gather_activations",
            "all_reduce_partial_outputs",
        ],
        "memory_model": {
            "dense_parameter_elements": dense_parameter_elements,
            "per_rank_parameter_elements": per_rank_parameter_elements,
            "parameter_memory_fraction_per_rank": round(per_rank_parameter_elements / dense_parameter_elements, 4),
            "activation_elements_per_rank": activation_elements_per_rank,
        },
        "throughput_model": {
            "compute_units": compute_units,
            "communication_bytes": communication_bytes,
            "communication_share": communication_share,
            "interpretation": "communication appears every block, unlike pure data parallel replication",
        },
        "relations": {
            "fsdp": "FSDP shards state but generally preserves full-layer compute semantics.",
            "pipeline": "Pipeline parallelism splits layer ranges; tensor parallelism splits inside a layer.",
        },
        "numerical_check": {
            "output_shape": shape(tp_output),
            "max_abs_diff_vs_dense": max_abs_diff(tp_output, dense_output),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
