from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
OBSERVED = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(UNIT_ROOT))
    except ValueError:
        return str(path)


def ensure_metrics() -> None:
    missing = [path for path in [SCRATCH, FRAMEWORK] if not path.exists()]
    if missing:
        names = ', '.join(_display(path) for path in missing)
        raise SystemExit(
            f'필수 metrics 파일이 없습니다: {names}.\n'
            '먼저 scratch_lab.py와 framework_lab.py를 실행하세요.\n'
            '예: python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py && '
            'python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/framework_lab.py'
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def require_keys(payload: dict[str, Any], keys: list[str], name: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SystemExit(f'metrics schema validation failed: {name} metrics missing keys {missing}')


def build_report(scratch: dict[str, Any], framework: dict[str, Any]) -> str:
    selected = framework['failure_triage']['selected_incident']
    validation = framework['recovery_decision']['post_resume_validation']
    return f'''# 09 Profiling, Monitoring, and Failure Recovery 실행 관측

## 관측 요약
- profile window: `{scratch['profile_window']['steps']}` steps, world size `{scratch['profile_window']['world_size']}`
- throughput baseline: `{scratch['throughput']['baseline_tokens_per_sec']}` tokens/sec
- throughput observed: `{scratch['throughput']['observed_tokens_per_sec']}` tokens/sec
- step time p50/p95: `{scratch['step_time_ms']['p50']}` ms / `{scratch['step_time_ms']['p95']}` ms
- peak allocated/reserved memory: `{scratch['memory_snapshot']['peak_allocated_mb']}` MB / `{scratch['memory_snapshot']['peak_reserved_mb']}` MB
- alerts: `{', '.join(scratch['alerts'])}`

## 병목 진단
- dominant bottleneck: `{scratch['dominant_bottleneck']}`
- communication_wait share: `{scratch['time_breakdown_pct']['communication_wait']}`%
- memory hint: `{scratch['memory_snapshot']['fragmentation_hint']}`
- heartbeat signal: `rank_2_heartbeat_lag` is treated as the first liveness split, not as a pure compute hotspot.

## Failure triage
- selected incident: `{selected['incident_id']}`
- classification: `{selected['classification']}`
- first checks: `{', '.join(selected['first_checks'])}`
- decision reason: `{framework['failure_triage']['decision_reason']}`

## Recovery decision
- action: `{framework['recovery_decision']['action']}`
- resume from: `{framework['recovery_decision']['resume_from']}`
- retry attempts: `{framework['retry_policy']['attempts_used']}/{framework['retry_policy']['max_attempts']}`
- checkpoint format: `{framework['checkpoint_manifest']['format']}`
- required checkpoint state: `{', '.join(framework['checkpoint_manifest']['required_state'])}`
- post-resume validation passed: `{validation['passed']}` over `{validation['validation_window_steps']}` steps

## 한국어 해석
- 평균 step time보다 p95와 second-half jitter가 먼저 커졌으므로, 단일 느린 함수보다 step lifecycle의 tail latency를 본다.
- throughput 하락과 rank heartbeat lag가 함께 움직였기 때문에 compute tuning 전에 rank-level liveness와 communication wait를 먼저 확인한다.
- reserved memory가 allocated보다 오래 높게 남는 신호는 즉시 batch size만 줄이는 대신 checkpoint/eval boundary와 allocator lifetime을 함께 보게 한다.
- recovery는 checkpoint 파일 존재 여부가 아니라 manifest, optimizer_state, scheduler_state, sampler_state, RNG, global_step 연속성 검증까지 포함한 run contract다.
'''


def run() -> str:
    ensure_metrics()
    scratch = load_json(SCRATCH)
    framework = load_json(FRAMEWORK)
    require_keys(
        scratch,
        ['profile_window', 'throughput', 'step_time_ms', 'time_breakdown_pct', 'memory_snapshot', 'alerts', 'dominant_bottleneck'],
        'scratch',
    )
    require_keys(
        framework,
        ['monitoring_contract', 'failure_triage', 'checkpoint_manifest', 'retry_policy', 'recovery_decision'],
        'framework',
    )
    report = build_report(scratch, framework)
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(report, encoding='utf-8')
    return report


if __name__ == '__main__':
    try:
        print(run())
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
