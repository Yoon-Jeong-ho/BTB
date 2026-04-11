from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCRATCH = UNIT_ROOT / "artifacts" / "scratch-manual" / "metrics.json"
DEFAULT_FRAMEWORK = UNIT_ROOT / "artifacts" / "framework-manual" / "metrics.json"
DEFAULT_OUTPUT = UNIT_ROOT / "artifacts" / "analysis-manual" / "latest_report.md"
DEFAULT_JSON = UNIT_ROOT / "artifacts" / "analysis-manual" / "observed_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic agentic training/eval loop artifacts.")
    parser.add_argument("--scratch-metrics", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--framework-metrics", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
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
            f"필수 metrics 파일이 없습니다: {names}.\n"
            "먼저 scratch_lab.py와 framework_lab.py를 실행하세요."
        )


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(scratch: dict[str, object], framework: dict[str, object]) -> dict[str, object]:
    iterations = scratch["iterations"]
    valid = [item for item in iterations if item["verifier"]["protocol_match"] and item["verifier"]["artifact_complete"]]
    rejected = [item for item in iterations if not item["verifier"]["protocol_match"] or not item["verifier"]["artifact_complete"]]
    final_gate = framework["gate_summary"]["final_gate"]
    dominant_risk = "benchmark_drift" if framework["benchmark_drift"]["observed_score"] >= framework["benchmark_drift"]["threshold"] else "retry_budget"
    return {
        "status": "runnable",
        "loop_id": scratch["loop_id"],
        "iteration_count": len(iterations),
        "valid_iteration_count": len(valid),
        "rejected_iteration_count": len(rejected),
        "final_decision": scratch["final_decision"]["action"],
        "final_gate": final_gate,
        "dominant_risk": dominant_risk,
        "benchmark_drift_score": framework["benchmark_drift"]["observed_score"],
        "retry_budget": scratch["experiment_contract"]["retry_budget"],
        "attempts_used": framework["retry_policy"]["attempts_used"],
        "role_sequence": framework["role_contract"]["separation_order"],
        "evidence_required_fields": framework["evidence_bundle"]["required_fields"],
    }


def render_report(summary: dict[str, object], scratch: dict[str, object], framework: dict[str, object]) -> str:
    final = scratch["final_decision"]
    gate = framework["gate_summary"]
    drift = framework["benchmark_drift"]
    roles = framework["role_contract"]
    required = ", ".join(framework["evidence_bundle"]["required_fields"][:6])
    return f"""# 03 Agentic Training and Eval Loops 실행 관측

## 요약

이 실행은 실제 학습이나 외부 서비스를 호출하지 않는 CPU-safe deterministic simulation이다. 목적은 agentic training/eval loop가 많이 반복되는지보다, 각 iteration이 같은 experiment contract 아래에서 planner / executor / verifier / critic 역할 분리를 남기고 stop/escalation rule을 적용하는지 관찰하는 것이다.

- loop id: `{summary['loop_id']}`
- iteration count: `{summary['iteration_count']}`
- valid comparable iterations: `{summary['valid_iteration_count']}`
- rejected / blocked iterations: `{summary['rejected_iteration_count']}`
- final gate: `{summary['final_gate']}`
- final decision: `{summary['final_decision']}`

## 역할 분리

역할 순서: `{ ' → '.join(roles['separation_order']) }`

- planner: experiment contract, bounded change set, retry budget, stop rule을 먼저 고정한다.
- executor: 승인된 change set만 실행하고 seed, config_hash, metric_json, artifact_manifest를 남긴다.
- verifier: protocol match, artifact completeness, baseline comparability, benchmark drift를 metric claim보다 먼저 확인한다.
- critic: verifier gate와 evidence refs를 인용해서 retry / rollback / stop / escalation 중 하나를 고른다.

Anti self-approval rule 중 핵심은 `{roles['anti_self_approval_rules'][0]}` 이다. 같은 agent가 계획과 승인을 동시에 담당하면 metric chasing과 self-justification이 빨라진다.

## Gate verdict

최종 gate는 `{gate['final_gate']}`이다. 이유: {gate['why']}

이번 loop에서 protocol match와 artifact completeness는 필수 gate였다. iteration 2는 점수가 좋아 보였지만 preprocessing cache가 바뀌고 artifact manifest가 비어 있어 rollback되었다. 따라서 verifier는 metric checker가 아니라 비교 가능성의 gatekeeper로 작동해야 한다.

## Evidence bundle

필수 evidence bundle 예시는 `{required}, ...` 이다. 이 bundle이 있어야 다음 iteration이 같은 사실을 다시 읽을 수 있다. metric json 하나만 남기면 config, split, failure slice, critic triage가 빠져서 주장 경계가 사라진다.

## Benchmark drift

관측 drift score는 `{drift['observed_score']}`이고 threshold는 `{drift['threshold']}`이다. dominant risk는 `{summary['dominant_risk']}`이다. long-tail slice regression과 drift probe warning이 같이 나타났으므로 더 많은 자동 retry 대신 benchmark/dataset contract review로 escalation한다.

## Stop / escalation

최종 action은 `{final['action']}`이다. reasons: `{', '.join(final['reasons'])}`. 이 단위의 핵심 학습은 agentic loop가 멈추지 않고 계속 도는 것이 아니라, stop rule과 escalation rule을 evidence bundle 위에서 적용하는 것이다.
"""


def analyze(
    scratch_path: Path = DEFAULT_SCRATCH,
    framework_path: Path = DEFAULT_FRAMEWORK,
    output_path: Path = DEFAULT_OUTPUT,
    json_output_path: Path = DEFAULT_JSON,
) -> dict[str, object]:
    ensure_metrics([scratch_path, framework_path])
    scratch = load_metrics(scratch_path)
    framework = load_metrics(framework_path)
    observed = summarize(scratch, framework)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(observed, scratch, framework), encoding="utf-8")
    json_output_path.write_text(json.dumps(observed, ensure_ascii=False, indent=2), encoding="utf-8")
    return observed


def main() -> int:
    args = parse_args()
    try:
        observed = analyze(args.scratch_metrics, args.framework_metrics, args.output, args.json_output)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(observed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
