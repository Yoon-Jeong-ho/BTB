# 02 Corpus, Tokenizer, and Data Mixture 회고

## 실행 후 바로 적어 보기
1. raw 문서 11개 중 trainable 문서가 7개로 줄어든 이유를 dedup과 contamination으로 나누어 설명해 보자.
2. `toy_unigram_like`와 `toy_aggressive_subword`의 평균 token 수 차이가 context window 64에서 어떤 비용 차이를 만들었는가?
3. contamination hit 2개를 제거하지 않았다면, 어떤 evaluation claim이 과장될 수 있는가?
4. domain token share를 보고 문서 수 기준 balance와 token budget 기준 balance가 어떻게 다를 수 있는지 적어 보자.
5. 한국어/영어/일본어가 같은 tokenizer를 공유할 때 어떤 언어가 더 잘게 쪼개졌고, 그 이유는 무엇이라고 해석할 수 있는가?

## 조금 더 깊게 생각하기
- dedup을 너무 강하게 하면 법률/코드/문서 템플릿처럼 반복되어야 하는 합법적 패턴도 사라질 수 있다. 어떤 guardrail을 추가할 수 있을까?
- 작은 고품질 domain corpus를 oversample하려면 contamination check와 validation split을 어떻게 더 보수적으로 설계해야 할까?
- 다음 DAPT 단위로 넘어가기 전에, 현재 mixture에서 더 늘리거나 줄이고 싶은 slice는 무엇이며 그 근거는 token share인가 품질인가?
