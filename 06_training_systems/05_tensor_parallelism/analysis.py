from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCRATCH = UNIT_ROOT / "artifacts" / "scratch_metrics.json"
DEFAULT_FRAMEWORK = UNIT_ROOT / "artifacts" / "framework_metrics.json"
DEFAULT_OUTPUT = UNIT_ROOT / "analysis.md"
OBSERVED = UNIT_ROOT / "artifacts" / "analysis_observed.json"


STABLE_REPORT = """# 05 Tensor Parallelism 분석

## Stable interpretation

Tensor parallelism is an intra-layer split: a single large layer is computed by
multiple ranks that each hold a matrix shard and an activation shard. This differs
from DDP-style replication and from FSDP/ZeRO-style state sharding because the
active matmul itself is partitioned.

## Korean-first reading

- column-parallel linear는 output feature 축을 나눠 rank별 activation shard를 만든다.
- row-parallel linear는 input feature 축과 weight row shard를 나눈 뒤 partial output을
  collective로 합친다.
- activation shard를 오래 유지하면 메모리와 bandwidth를 아낄 수 있지만, 다음
  연산이 같은 shard layout을 이해해야 한다.
- communication overhead는 메모리 절감의 대가다. 작은 CPU simulation에서도
  all-gather와 all-reduce가 어느 위치에 들어가는지 분리해 읽을 수 있다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic tensor parallel metrics.")
    parser.add_argument("--scratch-metrics", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--framework-metrics", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_metrics(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    def display(path: Path) -> str:
        try:
            return str(path.relative_to(UNIT_ROOT))
        except ValueError:
            return str(path)

    names = ", ".join(display(path) for path in missing)
    raise SystemExit(
        f"Missing required metrics file: {names}.\n"
        "Run scratch_lab.py and framework_lab.py first."
    )


def analyze(
    scratch_path: Path = DEFAULT_SCRATCH,
    framework_path: Path = DEFAULT_FRAMEWORK,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    ensure_metrics([scratch_path, framework_path])
    scratch = load_metrics(scratch_path)
    framework = load_metrics(framework_path)

    observed: dict[str, object] = {
        "status": "runnable",
        "stable_report": output_path.name,
        "observed_report": str(OBSERVED.relative_to(UNIT_ROOT)),
        "tp_world_size": scratch["tp_world_size"],
        "column_activation_shard": scratch["column_parallel"]["per_rank_activation_shape"],
        "row_activation_shard": scratch["row_parallel"]["per_rank_activation_shape"],
        "collectives_per_block": framework["collectives_per_block"],
        "scratch_estimated_communication_bytes": scratch["communication_overhead"]["estimated_bytes"],
        "framework_communication_share": framework["throughput_model"]["communication_share"],
        "max_abs_diff_vs_dense": max(
            scratch["max_abs_diff_vs_dense"],
            framework["numerical_check"]["max_abs_diff_vs_dense"],
        ),
        "interpretation": (
            "The simulation preserves dense math while exposing tensor/model parallel "
            "split points and per-block communication overhead."
        ),
    }

    output_path.write_text(STABLE_REPORT, encoding="utf-8")
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(json.dumps(observed, ensure_ascii=False, indent=2), encoding="utf-8")
    return observed


def main() -> int:
    args = parse_args()
    try:
        observed = analyze(args.scratch_metrics, args.framework_metrics, args.output)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(observed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
