# 07 Retrieval-Augmented Generation and Eval 이론 노트

이 문서는 Korean-first로 RAG를 “검색을 붙인 생성”이 아니라 **retriever-reader / retriever-generator split, retrieval grounding, context injection, citation/evidence expectation, failure mode, eval harness metrics**를 함께 설계하는 문제로 정리한다.

## 1. RAG intuition: parametric memory와 external evidence
- language model의 parametric memory는 일반 패턴과 과거 지식을 담지만 최신 정책, 사내 위키, 감사 로그처럼 출처와 시점이 중요한 지식에는 취약하다.
- retrieval-augmented generation은 답하기 전에 query와 관련된 chunk를 찾아 external evidence로 넣는 구조다.
- 핵심은 “검색했는가?”가 아니라 “검색된 evidence가 answer를 얼마나 제한했는가?”다.

## 2. retriever-reader vs retriever-generator
### retriever-reader
- retriever가 후보 chunk를 찾고 reader가 그 안에서 answer span 또는 evidence-rich answer를 읽는다.
- 근거 위치가 더 강하게 고정되므로 unsupported claim을 줄이기 쉽다.
- evidence가 없으면 abstain하거나 ask-back하는 규칙을 붙이기 좋다.

### retriever-generator
- retriever가 찾은 chunk를 context로 넣고 generator가 자연어 답변을 합성한다.
- multi-document summary와 synthesis에 유연하지만, 자유도가 커질수록 evidence 밖 연결 문장을 만들 수 있다.
- 따라서 retriever-generator에는 citation tag, source metadata, contradiction handling, unsupported claim checker가 필요하다.

## 3. retrieval grounding과 citation/evidence expectation
- retrieval grounding은 주요 claim이 retrieved evidence에 의해 support되는지의 문제다.
- citation은 grounding의 proxy일 수 있지만 grounding 자체는 아니다.
- 좋은 답변은 다음 기대를 만족해야 한다.
  - 주요 claim마다 대응 evidence가 있다.
  - citation은 관련 chunk나 section을 가리킨다.
  - evidence가 부족하면 uncertainty, abstain, ask-back 중 하나로 전환한다.
  - stale source와 newer source가 충돌하면 freshness metadata를 드러낸다.

## 4. context injection은 formatting이 아니라 behavior control
- chunk boundary, top-k, reranking, metadata, prompt order, citation tag가 모두 answer behavior를 바꾼다.
- chunk가 너무 짧으면 필요한 조건이 잘리고, 너무 길면 irrelevant context가 늘어난다.
- 오래된 source가 최신 source보다 위에 있으면 generator가 stale answer를 만들 수 있다.
- retrieved document 안의 명령문이나 formatting noise가 prompt처럼 작동하면 answer가 오염될 수 있다.

## 5. 대표 failure mode
- retrieval miss: relevant chunk가 top-k에 없다.
- missing evidence: query에 답할 근거가 corpus에 없다.
- stale source: 오래된 문서가 최신 문서보다 선택된다.
- irrelevant context: surface similarity가 높은 다른 문서가 answer input을 차지한다.
- unsupported claim: generator가 evidence 밖 내용을 그럴듯하게 추가한다.
- citation without support: citation은 있지만 claim을 실제로 support하지 않는다.
- over-confident answer: evidence가 부족한데도 abstain하지 않는다.

## 6. eval harness와 metrics 분리
### Retrieval metrics
- Recall@k, Precision@k, MRR, nDCG는 relevant evidence를 가져올 “기회”를 측정한다.
- retrieval metrics는 answer correctness를 보장하지 않는다.

### Answer / grounding metrics
- groundedness, faithfulness, citation precision, citation coverage, unsupported claim rate, abstention accuracy를 본다.
- fluency보다 evidence support를 분리해서 기록해야 한다.

### Online / product metrics
- acceptance proxy, correction rate, citation click, escalation, p50/p95 latency를 본다.
- offline eval harness가 좋아도 online user behavior와 어긋날 수 있다.

## 실행 결과 예시
`scratch_lab.py`는 `artifacts/scratch-manual/metrics.json`과 `rag_grounding_eval.svg`를 만든다. `framework_lab.py`는 deterministic lightweight retrieval/generation simulation으로 batch shape와 offline/online metrics를 기록한다. `analysis.py`는 두 metrics 파일이 없으면 실패하고, 있으면 `artifacts/analysis-manual/latest_report.md`에 실행 관측을 쓴다.

## Common Confusion
- RAG가 hallucination을 자동으로 없앤다고 믿는 실수
- citation count가 많으면 retrieval grounding도 높다고 믿는 실수
- retriever recall만 높이면 final answer quality도 오른다고 믿는 실수
- retriever-reader와 retriever-generator를 단순 모델 종류 차이로만 보는 실수
- 더 많은 context injection이 항상 좋은 답을 만든다고 믿는 실수
- offline metrics가 좋으면 online trust와 correction rate도 자동으로 좋아진다고 믿는 실수
