from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 04 Benchmark and Dataset Construction 분석

## 이 문서를 어떻게 읽을까
- 실행별 toy benchmark 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 benchmark/dataset construction을 해석하는 안정적인 프레임만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- benchmark는 leaderboard가 아니라 **task contract와 claim boundary를 고정하는 측정 계약**이다.
- dataset schema는 필드 목록만이 아니라 unit of record, source boundary, license tier, version freeze를 함께 포함해야 한다.
- source/split manifest는 random split보다 강한 hygiene를 요구한다. source와 template family가 split 사이를 건너면 leakage 위험이 커진다.
- annotation rubric과 QC는 label을 깨끗하게 보이게 하는 절차가 아니라 ambiguity와 disagreement를 기록하는 절차다.
- leakage, contamination, drift audit는 점수가 올랐을 때 그 점수를 capability improvement로 읽어도 되는지 확인하는 방어막이다.
- benchmark card, versioning, report template는 나중 연구 트랙이 숫자와 known limits를 함께 보고하게 만드는 운영 인터페이스다.

## 확인 질문
- task contract가 input/output/unit of record와 claim boundary를 명확히 고정하는가?
- dataset schema와 source/split manifest가 license, source, template family leakage를 같이 막는가?
- annotation rubric과 QC report가 agreement score뿐 아니라 major disagreement와 adjudication rule을 남기는가?
- contamination과 drift flag가 headline score 해석에 어떤 warning을 붙이는가?
- versioning 정책이 frozen core와 refresh slice를 구분해 과거 run과의 비교 가능성을 지키는가?

## 관련 이론
- [THEORY.md](./THEORY.md): benchmark card, task contract, dataset schema, source/split manifest, annotation rubric/QC, leakage/contamination/drift audit를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
'''


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def _ensure_metrics_exist() -> None:
    missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]
    if not missing:
        return
    missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
    raise SystemExit(
        '필수 metrics 파일이 없습니다: '
        f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'
    )


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)
    scratch_audit = scratch.get('leakage_contamination_drift_audit', {})
    framework_audit = framework.get('audit', {})
    split_manifest = framework.get('split_manifest', {})
    annotation = framework.get('annotation', {})
    qc = annotation.get('qc', {}) if isinstance(annotation, dict) else {}

    observed_report = f'''# 04 Benchmark and Dataset Construction 실행 관측

## 관측 결과
- benchmark card: `{scratch.get('benchmark_card', {}).get('benchmark_id', 'unknown')}`
- task contract unit: `{scratch.get('task_contract', {}).get('unit_of_record', 'unknown')}`
- dataset size: `{framework.get('dataset_size', 0)}`
- source/split manifest counts: `{split_manifest.get('counts', {})}`
- source disjoint: `{split_manifest.get('source_disjoint', False)}`
- template family disjoint: `{split_manifest.get('template_family_disjoint', False)}`
- annotation rubric dimensions: `{annotation.get('rubric_dimensions', []) if isinstance(annotation, dict) else []}`
- annotation QC agreement score: `{qc.get('agreement_score', 0)}`
- leakage exact cross-split hits: `{framework_audit.get('exact_cross_split_overlap_hits', 0)}`
- contamination flags: `{framework_audit.get('contamination_flags', 0)}`
- drift watchlist: `{framework_audit.get('drift_watchlist', [])}`
- versioning: `{framework.get('versioning', {})}`
- report template sections: `{framework.get('report_template', {}).get('sections', [])}`

## 한국어 해석
- 이 benchmark card는 `{scratch.get('benchmark_card', {}).get('primary_claim', 'unknown')}` claim만 허용하고, known non-goals를 함께 남긴다.
- task contract는 `agent_task_record`를 unit of record로 두어 response, citations, tool actions, refusal reason을 같은 사례 안에서 평가한다.
- dataset schema는 `record_id`, `source_id`, `split`, `slice_tags`, `license_tier` 같은 필수 필드로 재현 가능한 dataset contract를 만든다.
- source/split manifest에서 source disjoint=`{split_manifest.get('source_disjoint', False)}`, template family disjoint=`{split_manifest.get('template_family_disjoint', False)}`이므로 random split보다 강한 leakage 방어를 보여 준다.
- annotation rubric은 task_success, groundedness, policy_compliance를 분리하고, QC agreement `{qc.get('agreement_score', 0)}`와 adjudication rule을 함께 기록한다.
- leakage exact hit `{scratch_audit.get('exact_cross_split_overlap_hits', 0)}`, contamination flags `{scratch_audit.get('contamination_flags', 0)}`, drift watchlist `{scratch_audit.get('drift_watchlist', [])}`는 score 해석 전에 확인해야 할 warning이다.
- versioning과 report template는 frozen core benchmark와 refresh slice를 구분해 다음 연구 트랙의 비교 가능성을 보호한다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
