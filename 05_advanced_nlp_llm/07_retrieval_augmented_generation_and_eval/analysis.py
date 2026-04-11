from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 07 Retrieval-Augmented Generation and Eval 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy RAG 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 retriever-reader / retriever-generator split, retrieval grounding, context injection, citation/evidence expectation, failure mode, eval harness metrics를 읽는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- retriever-reader는 evidence span을 직접 읽거나 근거 부족 시 abstain하기 쉬워 unsupported claim을 줄이는 쪽에 강하다.
- retriever-generator는 여러 retrieved chunk를 fluent하게 합성하지만, context injection과 citation discipline이 약하면 evidence 밖 claim을 만들 수 있다.
- retrieval grounding은 citation 개수가 아니라 주요 claim이 retrieved evidence와 연결되는지로 판단한다.
- context injection에서는 chunk boundary, metadata, source freshness, citation tag, prompt order가 answer behavior를 바꾼다.
- eval harness는 retrieval metrics(recall@k, MRR, nDCG), answer metrics(groundedness, citation precision, unsupported claim rate), online metrics(acceptance, correction, citation click)을 분리해 읽어야 한다.

## 확인 질문
- retriever가 relevant chunk를 찾았는데도 generator가 unsupported claim을 만든다면 어떤 context injection 또는 prompt rule을 먼저 점검할 것인가?
- citation precision과 claim-level evidence coverage가 서로 어긋나는 사례는 무엇인가?
- retriever-reader가 답을 보류하고 retriever-generator가 추측하는 query는 어떤 failure mode를 보여 주는가?
- stale source가 top-k에 들어왔을 때 freshness metadata와 reranking은 어떻게 작동해야 하는가?
- offline retrieval recall이 높아도 online correction rate가 높게 남는다면 eval harness의 어느 층을 추가로 봐야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): retriever-reader/generator split, retrieval grounding, context injection, citation/evidence expectation, failure modes, eval harness를 다시 확인한다.
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

    scratch_retrieval = scratch.get('retrieval_metrics', {})
    scratch_grounding = scratch.get('grounding_eval', {})
    split_view = scratch.get('split_view', {})
    context = scratch.get('context_injection', {})
    failures = scratch.get('failure_modes', {})
    framework_retrieval = framework.get('retrieval_metrics', {})
    answer_metrics = framework.get('answer_metrics', {})
    probes = framework.get('failure_mode_probes', {})
    harness = framework.get('eval_harness', {})
    online = harness.get('online', {}) if isinstance(harness, dict) else {}

    observed_report = f'''# 07 Retrieval-Augmented Generation and Eval 실행 관측

## 관측 결과
- scratch retrieval metrics: `{scratch_retrieval}`
- scratch grounding eval: `{scratch_grounding}`
- split view: `{split_view}`
- context injection: `{context}`
- scratch failure modes: `{failures}`
- framework retrieval metrics: `{framework_retrieval}`
- framework answer metrics: `{answer_metrics}`
- framework failure probes: `{probes}`
- online proxy metrics: `{online}`

## 한국어 해석
- 이번 toy RAG 실험은 **retriever-reader**와 **retriever-generator**를 분리해 본다. reader-style은 근거 span을 읽거나 abstain하므로 unsupported claim을 줄이고, generator-style은 더 자연스럽게 합성하지만 citation과 evidence 규칙이 없으면 추측을 섞을 수 있다.
- retrieval grounding은 `{scratch_grounding.get('grounding_expectation', 'unknown')}`로 해석한다. 즉 citation이 붙었다는 사실보다 claim이 실제 retrieved evidence에 의해 support되는지를 본다.
- context injection은 `{context.get('prompt_order', [])}` 순서와 metadata/citation tag 포함 여부로 answer shape를 제한한다. stale source나 irrelevant context가 섞이면 generator가 오래된 근거를 채택할 수 있다.
- scratch에서는 primary watch가 `{failures.get('primary_watch', 'unknown')}`이고 framework에서도 highest risk가 `{probes.get('highest_risk', 'unknown')}`다. missing_evidence, stale_source, unsupported claim은 retrieval miss만으로 설명되지 않는 pipeline failure mode다.
- eval harness는 offline retrieval recall `{framework_retrieval.get('recall_at_3', 'unknown')}`, groundedness `{answer_metrics.get('groundedness', 'unknown')}`, online correction proxy `{online.get('correction_rate_proxy', 'unknown')}`를 따로 본다. retrieval metric이 좋아도 answer grounding과 사용자 correction metric은 따로 흔들릴 수 있다.

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
