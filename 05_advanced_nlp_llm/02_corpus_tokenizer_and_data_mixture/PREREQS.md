# 02 Corpus, Tokenizer, and Data Mixture 선행 개념

## 꼭 알고 오면 좋은 것
- tokenization / subword / token id / embedding의 기본 흐름
- next-token language modeling이 무엇을 학습하는지에 대한 기본 감각
- train / validation / test split과 leakage가 왜 문제인지에 대한 이해
- sequence length, context window, truncation이 학습 비용과 정보량에 어떤 영향을 주는지
- 데이터 분포 차이, class/domain imbalance, sample weighting 같은 기본 데이터 관점
- 평균, 중앙값, percentile, 비율처럼 corpus 통계를 읽는 최소한의 감각

## 빠른 자기 점검
- 같은 문서 집합이라도 tokenizer를 바꾸면 왜 문서당 token 수와 context window 사용 방식이 달라지는지 설명할 수 있는가?
- "문서 수 기준으로는 균형인데 token 수 기준으로는 불균형"인 상황을 예로 들 수 있는가?
- deduplication과 contamination check가 왜 같은 작업이 아닌지 한두 문장으로 구분할 수 있는가?
- multilingual corpus에서 shared tokenizer가 특정 언어에 불리할 수 있는 이유를 말할 수 있는가?
- 작은 고품질 도메인 corpus를 oversample하는 것이 왜 도움이 되기도 하고 과적합/편향을 만들기도 하는지 설명할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — causal / masked / span corruption objective 차이를 먼저 정리한다.
