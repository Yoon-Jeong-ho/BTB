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
            '예: python3 07_frontier_labs/05_open_ended_research_tracks/scratch_lab.py && '
            'python3 07_frontier_labs/05_open_ended_research_tracks/framework_lab.py'
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def require_keys(payload: dict[str, Any], keys: list[str], name: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SystemExit(f'metrics schema validation failed: {name} metrics missing keys {missing}')


def build_report(scratch: dict[str, Any], framework: dict[str, Any]) -> str:
    scope = scratch['research_scope']
    decisions = framework['decision_summary']['decision_counts']
    result_rules = framework['decision_by_result_type']
    return f'''# 05 Open-Ended Research Tracks 실행 관측

## 연구 범위 / research scope
- track id: `{scratch['track_id']}`
- research scope: `{scope['research scope']}`
- north-star question: `{scope['north-star question']}`
- 이번 iteration focus: `{', '.join(scope['this_iteration_focus'])}`
- out-of-scope: `{', '.join(scope['out_of_scope'])}`

## hypothesis registry와 iteration boundary
- hypothesis registry size: `{len(scratch['hypothesis_registry']['hypotheses'])}`
- total budgeted runs: `{scratch['iteration_boundary_summary']['total_budgeted_runs']}`
- changed variables: `{', '.join(scratch['iteration_boundary_summary']['changed_variables'])}`
- 모든 hypothesis는 iteration boundary, kill criteria, evidence standard, reopen condition을 함께 가진다.

## evidence standard
- required fields: `{', '.join(scratch['evidence_standard']['required_fields'])}`
- negative result: `{scratch['evidence_standard']['negative_vs_inconclusive_rule']['negative result']}`
- inconclusive result: `{scratch['evidence_standard']['negative_vs_inconclusive_rule']['inconclusive result']}`
- exploratory research에서도 baseline-relative signal, failure slice notes, qualitative evidence, negative result log를 같이 남긴다.

## stop / pause / escalate / archive 결정
- stop decisions: `{decisions['stop']}`
- pause decisions: `{decisions['pause']}`
- escalate decisions: `{decisions['escalate']}`
- archive decisions: `{decisions['archive']}`
- negative result → `{result_rules['negative result']}`
- inconclusive result → `{result_rules['inconclusive result']}`
- trust failure → `{result_rules['trust failure']}`
- success stop → `{result_rules['success stop']}`

## reopen condition
- archive contract: `{framework['archive_contract']['anti_wandering_guard']}`
- reopen condition examples:
{chr(10).join(f"  - {condition}" for condition in framework['archive_contract']['reopen condition'])}

## 한국어 해석
- open-ended research는 범위 없는 탐색이 아니라, research scope와 north-star question을 작게 자르는 운영 문제다.
- hypothesis registry는 아이디어 목록이 아니라 claim, mechanism, iteration boundary, kill criteria, evidence standard를 묶어 keep/kill 판단을 가능하게 하는 장치다.
- negative result와 inconclusive result는 같은 실패가 아니다. negative result는 충분한 증거로 현재 가설을 접는 archive 결정이고, inconclusive result는 측정/범위 신뢰가 약해서 pause 후 재측정하는 결정이다.
- stop / pause / escalate / archive decision을 명시하면 끈질긴 탐색과 scope creep를 구분할 수 있고, reopen condition까지 남기면 다음 사람이 같은 질문을 더 싸고 정직하게 다시 열 수 있다.

## 관련 이론
- [THEORY.md](./THEORY.md) — open-ended research 운영 원리
- [PREREQS.md](./PREREQS.md) — 선행 개념과 자기 점검
- [reflection.md](./reflection.md) — 실행 전후 연구 운영 회고
'''


def run() -> str:
    ensure_metrics()
    scratch = load_json(SCRATCH)
    framework = load_json(FRAMEWORK)
    require_keys(
        scratch,
        ['track_id', 'research_scope', 'hypothesis_registry', 'iteration_boundary_summary', 'evidence_standard', 'evidence_log'],
        'scratch',
    )
    require_keys(
        framework,
        ['operation_contract', 'decision_by_result_type', 'decision_log', 'decision_summary', 'archive_contract'],
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
