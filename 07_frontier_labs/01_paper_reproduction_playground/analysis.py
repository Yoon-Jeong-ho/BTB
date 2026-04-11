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
            '예: python3 07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py && '
            'python3 07_frontier_labs/01_paper_reproduction_playground/framework_lab.py'
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def require_keys(payload: dict[str, Any], keys: list[str], name: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SystemExit(f'metrics schema validation failed: {name} metrics missing keys {missing}')


def _comparison_line(claim_id: str, row: dict[str, Any]) -> str:
    metric = str(row['metric'])
    baseline = row['baseline'][metric]
    reported = row['reported'][metric]
    reproduced = row['reproduced'][metric]
    return (
        f'- `{claim_id}` `{metric}`: baseline `{baseline}`, reported `{reported}`, reproduced `{reproduced}`; '
        f'delta_vs_baseline `{row["delta_vs_baseline"]}`, delta_vs_reported `{row["delta_vs_reported"]}`'
    )


def build_report(scratch: dict[str, Any], framework: dict[str, Any]) -> str:
    matrix_lines = '\n'.join(
        f'- `{row["claim_id"]}`: {row["decision"]} — {row["observed_signal"]}'
        for row in scratch['claim_evidence_matrix']
    )
    comparison_lines = '\n'.join(_comparison_line(claim_id, row) for claim_id, row in scratch['comparisons'].items())
    hypotheses = '\n'.join(
        f'- `{item["hypothesis_id"]}`: {item["evidence"]} → next `{item["next_check"]}`'
        for item in scratch['mismatch_hypotheses']
    )
    hygiene = scratch['artifact_hygiene']
    return f'''# 01 Paper Reproduction Playground 실행 관측

## claim/evidence matrix
{matrix_lines}

## baseline / reported / reproduced 비교
{comparison_lines}

## scope control
- principle: `{scratch['scope_control']['principle']}`
- claim scope: `{scratch['scope_control']['claim_scope']}`
- dataset scope: `{scratch['scope_control']['dataset_scope']}`
- allowed claim: `{scratch['scope_control']['allowed_claim']}`
- not allowed claim: `{scratch['scope_control']['not_allowed_claim']}`
- framework primary comparison: `{framework['comparison_policy']['primary_comparison']}`

## variance와 mismatch hypothesis
- seed count: `{scratch['variance_summary']['seed_count']}`
- accuracy mean/std: `{scratch['variance_summary']['accuracy_mean']}` / `{scratch['variance_summary']['accuracy_std']}`
{hypotheses}

## artifact hygiene
- ready for handoff: `{hygiene['ready_for_handoff']}`
- missing required artifacts: `{hygiene['missing_required_artifacts']}`
- manifest missing: `{framework['artifact_manifest']['missing']}`
- generated figure: `{scratch['artifacts']['figure']}`

## 한국어 해석
- 이 결과는 논문 전체 복제가 아니라 claim/evidence matrix를 이용한 reduced claim 재현 계약이다.
- baseline, reported, reproduced 숫자는 한 줄에 같이 놓되, primary comparison은 같은 protocol의 reproduced baseline vs reproduced method로 제한한다.
- reported gap과 reproduced gap이 다르면 먼저 preprocessing_alignment, seed_variance, budget_mismatch 같은 mismatch hypothesis를 기록하고, 구현 성공/실패를 단정하지 않는다.
- artifact hygiene는 다음 사람이 scope boundary, claim/evidence matrix, variance summary, mismatch hypotheses를 읽고 바로 다음 run을 설계할 수 있게 만드는 최소 조건이다.

## 관련 이론
- [THEORY.md](./THEORY.md)
- [PREREQS.md](./PREREQS.md)
'''


def run() -> str:
    ensure_metrics()
    scratch = load_json(SCRATCH)
    framework = load_json(FRAMEWORK)
    require_keys(
        scratch,
        ['claim_evidence_matrix', 'comparisons', 'scope_control', 'variance_summary', 'mismatch_hypotheses', 'artifact_hygiene', 'artifacts'],
        'scratch',
    )
    require_keys(
        framework,
        ['comparison_policy', 'artifact_manifest', 'reproduction_decision'],
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
