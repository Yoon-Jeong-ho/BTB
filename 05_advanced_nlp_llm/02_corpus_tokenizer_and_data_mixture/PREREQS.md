# 02 Corpus, Tokenizer, and Data Mixture 선행 개념

## 꼭 알고 오면 좋은 것
- tokenization, subword, token id, embedding의 기본 흐름
- next-token language modeling이 token stream을 학습한다는 감각
- train / validation / test split과 leakage가 왜 위험한지
- sequence length, context window, truncation이 비용과 정보량에 미치는 영향
- 데이터 분포 차이, domain imbalance, sample weighting의 기본 개념
- 평균, 비율, token share 같은 작은 corpus 통계를 읽는 방법

## 빠른 자기 점검
- 같은 문서라도 tokenizer를 바꾸면 왜 문서당 token 수가 달라지는지 설명할 수 있는가?
- exact duplicate와 near duplicate를 어떻게 구분할 수 있는가?
- contamination이 평가 지표를 왜 부풀릴 수 있는지 말할 수 있는가?
- 문서 수 기준 balance와 token budget 기준 balance가 다를 수 있는 예를 들 수 있는가?
- multilingual corpus에서 shared tokenizer가 특정 언어를 더 잘게 쪼갤 수 있는 이유를 말할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — language modeling objective가 token stream을 어떻게 소비하는지 먼저 정리한다.
