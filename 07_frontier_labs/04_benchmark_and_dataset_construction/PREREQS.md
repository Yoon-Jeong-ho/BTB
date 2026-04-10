# 04 Benchmark and Dataset Construction 선행 개념

## 꼭 알고 오면 좋은 것
- baseline, metric, acceptance gate를 문장으로 먼저 고정해야 실험 비교가 정직해진다는 감각
- train / dev / test split과 leakage, contamination, dedup이 왜 서로 다른 문제인지에 대한 이해
- annotation rubric, label taxonomy, disagreement, adjudication이 dataset 품질을 좌우한다는 기본 인식
- class imbalance, slice evaluation, long-tail failure처럼 데이터 분포를 읽는 최소한의 감각
- benchmark 점수 하나보다 reporting context와 known limits가 함께 있어야 해석이 가능하다는 문제의식
- dataset versioning과 benchmark refresh가 비교 가능성에 어떤 긴장을 만드는지에 대한 기본 이해

## 먼저 다시 보면 좋은 단위
- [07_frontier_labs/03_agentic_training_and_eval_loops](../03_agentic_training_and_eval_loops/README.md) — verifier gate와 benchmark drift 관찰 포인트를 benchmark 설계 관점으로 다시 읽는다.
- [07_frontier_labs/02_capstone_model_building](../02_capstone_model_building/README.md) — problem statement, dataset contract, eval contract를 한 문서에서 묶는 감각을 복습한다.
- [05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture](../../05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/README.md) — dedup, contamination, token/data mixture 관점을 benchmark construction과 연결해 본다.
- [05_advanced_nlp_llm/08_alignment_safety_and_model_behavior](../../05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/README.md) — behavioral benchmark와 rubric 설계가 capability score와 어떻게 다른지 다시 확인한다.
- [01_ml/03_model_selection_and_interpretation](../../01_ml/03_model_selection_and_interpretation/README.md) — validation split과 metric 해석, 잘못된 비교선이 만드는 착시를 기본 ML 관점에서 복습한다.

## 빠른 자기 점검
- benchmark를 leaderboard가 아니라 측정 계약이라고 부르는 이유를 한두 문장으로 설명할 수 있는가?
- dataset contract에서 schema 외에 source boundary, unit of record, version freeze를 왜 같이 적어야 하는지 말할 수 있는가?
- random split이 충분하지 않은 leakage 사례를 user/source/time/template 관점에서 예로 들 수 있는가?
- annotator disagreement가 단순 noise가 아니라 task ambiguity 신호일 수 있다는 점을 설명할 수 있는가?
- contamination, leakage, benchmark gaming이 각각 무엇을 뜻하고 어떤 audit가 필요한지 구분할 수 있는가?
- benchmark를 refresh해야 할 때와 freeze를 유지해야 할 때의 trade-off를 비교해 말할 수 있는가?
