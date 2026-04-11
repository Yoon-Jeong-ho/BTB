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


STABLE_REPORT = """# 08 Hybrid Parallel Topologies 분석

## Stable interpretation

Hybrid parallel topology planning is a model-hardware placement problem, not a
checklist that turns on every parallelism option. Data parallelism owns the
replica/batch axis, tensor parallelism owns latency-sensitive intra-layer
collectives, pipeline parallelism owns stage/time scheduling, and FSDP/ZeRO-style
state sharding owns state residency plus checkpoint lifecycle.

## Korean-first reading

- data parallel 축은 global/effective batch와 gradient sync cadence를 담당한다.
- tensor parallel 축은 레이어 내부 matmul/head split을 만들며, all-reduce/all-gather가
  자주 발생하므로 빠른 node-local link 위에 두는 편이 안전하다.
- pipeline parallel 축은 stage boundary와 microbatch schedule을 만들고, activation
  send/recv와 bubble/load-balance 위험을 남긴다.
- FSDP/state sharding 축은 parameter, gradient, optimizer state의 resident memory와
  checkpoint save/load 계약을 바꾼다.
- 좋은 hybrid topology는 memory fit, communication tradeoff, bottleneck reasoning,
  checkpoint portability를 동시에 읽을 수 있어야 한다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic hybrid parallel topology metrics.")
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
    preferred = str(scratch["preferred_candidate"])
    preferred_candidate = next(
        candidate for candidate in scratch["candidate_topologies"] if candidate["name"] == preferred
    )

    observed: dict[str, object] = {
        "status": "runnable",
        "stable_report": output_path.name,
        "observed_report": str(OBSERVED.relative_to(UNIT_ROOT)),
        "preferred_candidate": preferred,
        "axis_product": preferred_candidate["axis_product"],
        "parallel_axes": list(framework["device_mesh_axes"]),
        "memory_margin_gb": preferred_candidate["memory_budget"]["memory_margin_gb"],
        "communication_hotspots": preferred_candidate["communication_budget"]["communication_hotspots"],
        "primary_risk": preferred_candidate["communication_budget"]["primary_risk"],
        "rank_mesh_contract": framework["rank_mesh_contract"],
        "bottleneck_reasoning": framework["bottleneck_reasoning"],
        "topology_lesson": (
            "Keep tensor-parallel traffic on fast links, use pipeline/FSDP to fit the model, "
            "and keep DP/FSDP checkpoint metadata explicit enough for recovery."
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
