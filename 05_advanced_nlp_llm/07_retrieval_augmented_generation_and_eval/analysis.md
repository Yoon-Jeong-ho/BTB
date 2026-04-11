# 07 Retrieval-Augmented Generation and Eval 분석

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
