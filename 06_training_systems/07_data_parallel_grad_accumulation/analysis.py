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


STABLE_REPORT = """# 07 Data Parallel + Grad Accumulation 분석

## Stable interpretation

Data parallelism expands the batch axis: every rank keeps a full model replica,
processes a different data shard, and then participates in gradient all-reduce.
Grad accumulation changes optimizer step cadence: several microsteps contribute
to one optimizer update, so effective batch is local batch × world size ×
accumulation steps.

## Korean-first reading

- data parallel은 모델을 복제하고 batch shard를 나누는 축이다. 모델 내부 연산을
  쪼개는 tensor parallel이나 stage를 나누는 pipeline parallel과 구분한다.
- grad accumulation은 local batch를 그대로 둔 채 optimizer step을 늦춰 effective
  batch를 키우는 스케줄링 정책이다.
- accumulation window 안에서는 `no_sync`/deferred sync로 all-reduce 횟수를 줄일 수
  있지만, boundary에서는 여전히 gradient synchronization 계약을 지켜야 한다.
- loss normalization은 microstep loss를 accumulation step 수로 나눠 gradient scale을
  맞추는 장치다.
- gradient clipping과 scheduler step은 microstep마다가 아니라 accumulation boundary의
  optimizer step 직전에/직후에 해석해야 한다.
- 같은 effective batch라도 큰 local batch와 작은 local batch + accumulation은
  activation memory, kernel efficiency, throughput trace가 다르게 보인다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 안정 해석 프레임이며, 실행별
숫자는 observed JSON을 확인한다.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic data parallel + grad accumulation metrics.")
    parser.add_argument("--scratch-metrics", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--framework-metrics", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(UNIT_ROOT))
    except ValueError:
        return str(path)


def ensure_metrics(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        names = ", ".join(display_path(path) for path in missing)
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
        "scratch_effective_batch": scratch["effective_batch_per_optimizer_step"],
        "framework_effective_batch": framework["effective_batch_per_optimizer_step"],
        "scratch_optimizer_steps": scratch["optimizer_step_count"],
        "framework_optimizer_steps": framework["optimizer_step_cadence"]["optimizer_steps"],
        "scratch_deferred_sync_calls": scratch["sync_policy_comparison"]["deferred_sync_all_reduce_count"],
        "scratch_every_step_sync_calls": scratch["sync_policy_comparison"]["every_step_all_reduce_count"],
        "framework_deferred_sync_calls": framework["communication_model"]["deferred_sync_calls"],
        "loss_normalization": scratch["loss_normalization"]["scale_per_microstep"],
        "gradient_clipping": framework["optimizer_dynamics"]["gradient_clipping"],
        "memory_tradeoff": scratch["memory_model_mb"]["interpretation"],
        "interpretation": (
            "Data parallel and grad accumulation both increase batch budget, but one adds rank parallelism "
            "while the other delays optimizer cadence under a local memory ceiling."
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
