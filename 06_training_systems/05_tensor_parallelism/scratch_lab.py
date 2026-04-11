from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


UNIT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = UNIT_ROOT / "artifacts"
METRICS_PATH = ARTIFACT_DIR / "scratch_metrics.json"
SVG_PATH = ARTIFACT_DIR / "tensor_parallelism_shards.svg"
FP32_BYTES = 4


Matrix = list[list[int]]


def shape(matrix: Matrix) -> list[int]:
    if not matrix:
        return [0, 0]
    return [len(matrix), len(matrix[0])]


def make_matrix(rows: int, cols: int, *, offset: int = 0, modulus: int = 17) -> Matrix:
    return [
        [((row + 1) * (col + 3 + offset)) % modulus - modulus // 2 for col in range(cols)]
        for row in range(rows)
    ]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        return []
    inner = len(right)
    if any(len(row) != inner for row in left):
        raise ValueError("left columns must equal right rows")
    cols = len(right[0])
    return [
        [sum(left_row[k] * right[k][col] for k in range(inner)) for col in range(cols)]
        for left_row in left
    ]


def split_columns(matrix: Matrix, world_size: int) -> list[Matrix]:
    cols = len(matrix[0])
    if cols % world_size != 0:
        raise ValueError("column count must divide world_size")
    shard_width = cols // world_size
    return [
        [row[rank * shard_width : (rank + 1) * shard_width] for row in matrix]
        for rank in range(world_size)
    ]


def split_rows(matrix: Matrix, world_size: int) -> list[Matrix]:
    rows = len(matrix)
    if rows % world_size != 0:
        raise ValueError("row count must divide world_size")
    shard_height = rows // world_size
    return [matrix[rank * shard_height : (rank + 1) * shard_height] for rank in range(world_size)]


def concat_columns(shards: Iterable[Matrix]) -> Matrix:
    shard_list = list(shards)
    rows = len(shard_list[0])
    return [
        [value for shard in shard_list for value in shard[row_index]]
        for row_index in range(rows)
    ]


def sum_matrices(matrices: Iterable[Matrix]) -> Matrix:
    matrix_list = list(matrices)
    rows, cols = shape(matrix_list[0])
    return [
        [sum(matrix[row][col] for matrix in matrix_list) for col in range(cols)]
        for row in range(rows)
    ]


def max_abs_diff(left: Matrix, right: Matrix) -> int:
    rows, cols = shape(left)
    return max(abs(left[row][col] - right[row][col]) for row in range(rows) for col in range(cols))


def checksum(matrix: Matrix) -> int:
    return sum(sum(row) for row in matrix)


def write_svg() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300">
  <rect width="760" height="300" fill="#fff7ed"/>
  <text x="24" y="36" font-family="monospace" font-size="18" fill="#7c2d12">Tensor parallel shard map (CPU simulation)</text>
  <text x="48" y="78" font-family="monospace" font-size="13" fill="#111827">Column-parallel: split output features, keep activation shards</text>
  <rect x="48" y="94" width="130" height="54" fill="#dbeafe" stroke="#1d4ed8"/>
  <rect x="188" y="94" width="130" height="54" fill="#bfdbfe" stroke="#1d4ed8"/>
  <rect x="328" y="94" width="130" height="54" fill="#93c5fd" stroke="#1d4ed8"/>
  <rect x="468" y="94" width="130" height="54" fill="#60a5fa" stroke="#1d4ed8"/>
  <text x="74" y="126" font-family="monospace" font-size="12">rank0 [8,4]</text>
  <text x="214" y="126" font-family="monospace" font-size="12">rank1 [8,4]</text>
  <text x="354" y="126" font-family="monospace" font-size="12">rank2 [8,4]</text>
  <text x="494" y="126" font-family="monospace" font-size="12">rank3 [8,4]</text>
  <text x="48" y="188" font-family="monospace" font-size="13" fill="#111827">Row-parallel: split input features, all-reduce partial outputs</text>
  <rect x="48" y="204" width="130" height="54" fill="#dcfce7" stroke="#15803d"/>
  <rect x="188" y="204" width="130" height="54" fill="#bbf7d0" stroke="#15803d"/>
  <rect x="328" y="204" width="130" height="54" fill="#86efac" stroke="#15803d"/>
  <rect x="468" y="204" width="130" height="54" fill="#4ade80" stroke="#15803d"/>
  <text x="72" y="236" font-family="monospace" font-size="12">partial [3,6]</text>
  <text x="212" y="236" font-family="monospace" font-size="12">partial [3,6]</text>
  <text x="352" y="236" font-family="monospace" font-size="12">partial [3,6]</text>
  <text x="492" y="236" font-family="monospace" font-size="12">partial [3,6]</text>
  <line x1="610" y1="231" x2="690" y2="231" stroke="#111827" stroke-width="2"/>
  <text x="610" y="217" font-family="monospace" font-size="12">all_reduce_sum</text>
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")


def run() -> dict[str, object]:
    tp_world_size = 4
    input_activation = make_matrix(3, 8, offset=1)
    column_weight = make_matrix(8, 16, offset=5)
    row_weight = make_matrix(16, 6, offset=9)

    column_weight_shards = split_columns(column_weight, tp_world_size)
    column_activation_shards = [matmul(input_activation, shard) for shard in column_weight_shards]
    gathered_column_activation = concat_columns(column_activation_shards)
    dense_column_activation = matmul(input_activation, column_weight)

    row_input_shards = split_columns(gathered_column_activation, tp_world_size)
    row_weight_shards = split_rows(row_weight, tp_world_size)
    row_partial_outputs = [
        matmul(input_shard, weight_shard)
        for input_shard, weight_shard in zip(row_input_shards, row_weight_shards)
    ]
    reduced_row_output = sum_matrices(row_partial_outputs)
    dense_output = matmul(dense_column_activation, row_weight)

    column_gather_elements = sum(len(row) for shard in column_activation_shards for row in shard)
    row_reduce_elements = sum(len(row) for partial in row_partial_outputs for row in partial)
    estimated_bytes = (column_gather_elements + row_reduce_elements) * FP32_BYTES

    metrics: dict[str, object] = {
        "status": "runnable",
        "tp_world_size": tp_world_size,
        "input_shape": shape(input_activation),
        "column_parallel": {
            "global_weight_shape": shape(column_weight),
            "per_rank_weight_shape": shape(column_weight_shards[0]),
            "per_rank_activation_shape": shape(column_activation_shards[0]),
            "collective": "all_gather_if_full_activation_required",
            "reconstructed_activation_shape": shape(gathered_column_activation),
        },
        "row_parallel": {
            "global_weight_shape": shape(row_weight),
            "per_rank_weight_shape": shape(row_weight_shards[0]),
            "per_rank_activation_shape": shape(row_input_shards[0]),
            "per_rank_partial_output_shape": shape(row_partial_outputs[0]),
            "collective": "all_reduce_sum",
            "reduced_output_shape": shape(reduced_row_output),
        },
        "rank_summaries": [
            {
                "rank": rank,
                "column_weight_shape": shape(column_weight_shards[rank]),
                "column_activation_shape": shape(column_activation_shards[rank]),
                "row_input_shape": shape(row_input_shards[rank]),
                "row_weight_shape": shape(row_weight_shards[rank]),
                "row_partial_output_shape": shape(row_partial_outputs[rank]),
                "activation_checksum": checksum(column_activation_shards[rank]),
                "partial_output_checksum": checksum(row_partial_outputs[rank]),
            }
            for rank in range(tp_world_size)
        ],
        "communication_overhead": {
            "column_all_gather_elements": column_gather_elements,
            "row_all_reduce_elements": row_reduce_elements,
            "estimated_bytes": estimated_bytes,
            "note": "CPU-only size model; no real interconnect is used.",
        },
        "max_abs_diff_vs_dense": max_abs_diff(reduced_row_output, dense_output),
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(UNIT_ROOT)),
            "svg": str(SVG_PATH.relative_to(UNIT_ROOT)),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_svg()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
