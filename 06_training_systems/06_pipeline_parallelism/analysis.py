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


STABLE_REPORT = """# 06 Pipeline Parallelism 분석

## Stable interpretation

Pipeline parallelism is execution-path partitioning: a sequential model is split
into pipeline stages, and microbatches move across those stages over time. It is
not a real multi-device runtime in this unit; the labs use deterministic CPU
simulation to make schedule, bubble, throughput, activation transfer, and
partition-balance trade-offs visible.

## Korean-first reading

- pipeline stage는 레이어 묶음을 맡는 실행 구간이며, stage boundary는 activation
  transfer 계약을 만든다.
- microbatch schedule은 warmup / steady / cooldown을 만들고, fill/drain 구간의
  idle slot이 pipeline bubble로 관찰된다.
- microbatch 수가 늘면 bubble fraction은 줄어들 수 있지만, transfer 횟수와
  bookkeeping도 함께 늘어난다.
- 1F1B는 backward를 더 빨리 시작해 GPipe식 all-forward-then-all-backward보다
  activation 보관량을 낮추는 방향의 schedule policy다.
- partition은 레이어 개수 균등 분할이 아니라 stage별 compute, memory,
  communication payload를 함께 맞추는 문제다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic pipeline parallel metrics.")
    parser.add_argument("--scratch-metrics", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--framework-metrics", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(UNIT_ROOT))
    except ValueError:
        return str(path)


def ensure_metrics(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        names = ", ".join(_display_path(path) for path in missing)
        raise SystemExit(
            f"Missing required metrics file: {names}.\n"
            "Run scratch_lab.py and framework_lab.py first."
        )


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "scratch_policy": scratch["schedule_summary"]["policy"],
        "framework_policy": framework["schedule_policy"],
        "num_stages": {
            "scratch": scratch["num_stages"],
            "framework": framework["num_stages"],
        },
        "microbatches": {
            "scratch": scratch["microbatches"],
            "framework": framework["microbatches"],
        },
        "bubble_fraction": {
            "scratch_forward_fill_drain": scratch["schedule_summary"]["bubble_fraction"],
            "framework_1f1b": framework["schedule_metrics"]["bubble_fraction"],
        },
        "throughput_microbatches_per_slot": {
            "scratch_forward_fill_drain": scratch["schedule_summary"]["throughput_microbatches_per_slot"],
            "framework_1f1b": framework["schedule_metrics"]["throughput_microbatches_per_slot"],
        },
        "activation_transfer_bytes": {
            "scratch_forward_only": scratch["activation_transfer"]["estimated_bytes"],
            "framework_forward_backward": framework["activation_transfer_model"]["estimated_bytes"],
        },
        "partition_balance": {
            "scratch_max_over_min_stage_compute": scratch["partition_balance"]["max_over_min_stage_compute"],
            "framework_max_over_min_stage_compute": framework["partitioning_concerns"]["max_over_min_stage_compute"],
        },
        "memory_tradeoff": {
            "gpipe_peak_saved_microbatches": framework["activation_memory_model"]["gpipe_peak_saved_microbatches"],
            "one_f1b_peak_saved_microbatches": framework["activation_memory_model"]["one_f1b_peak_saved_microbatches"],
        },
        "interpretation": (
            "More microbatches reduce visible fill/drain bubbles, but stage imbalance "
            "and activation transfers still bound pipeline throughput."
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
