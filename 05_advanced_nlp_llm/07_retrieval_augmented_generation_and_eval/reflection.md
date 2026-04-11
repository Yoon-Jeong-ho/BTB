# 07 Retrieval-Augmented Generation and Eval 회고 질문

이 reflection은 Korean-first로 작성한다. 답변할 때는 toy metrics 숫자를 그대로 베끼기보다, retriever-reader / retriever-generator split, retrieval grounding, context injection, citation/evidence expectation, failure mode, eval harness metrics를 연결해서 설명한다.

## 학습자 프롬프트
1. 이번 실습에서 retriever-reader가 retriever-generator보다 unsupported claim을 줄이기 쉬웠던 이유를 evidence 사용 방식으로 설명하라.
2. citation이 붙은 답변도 retrieval grounding이 부족할 수 있는 사례를 하나 만들고, 어떤 evidence check가 필요한지 적어라.
3. context injection에서 source metadata와 citation tag를 빼면 어떤 failure mode가 먼저 커질지 예측하라.
4. missing evidence query에서 generator가 추측하고 reader가 abstain하는 차이를 answer / abstain / ask-back 정책 관점으로 설명하라.
5. recall@3, MRR, nDCG 같은 retrieval metrics가 높아도 groundedness나 unsupported claim rate가 나쁠 수 있는 이유를 적어라.
6. offline eval harness와 online acceptance/correction/citation-click metrics가 서로 어긋날 때 어떤 추가 로그를 수집할지 제안하라.
7. stale source가 top-k에 함께 들어온 경우, freshness metadata와 reranking을 어떻게 바꾸면 좋을지 설계하라.
8. 다음 단위 alignment/safety 관점에서 RAG 답변의 uncertainty expression과 trust calibration을 어떻게 평가할지 적어라.

## 제출 체크리스트
- retriever-reader와 retriever-generator를 둘 다 언급했다.
- retrieval grounding과 citation/evidence expectation을 구분했다.
- context injection 선택이 failure mode를 만든다는 점을 설명했다.
- eval harness metrics를 retrieval, answer grounding, online product 층으로 나눴다.
- unsupported claim, missing evidence, stale source 중 최소 두 가지 failure mode를 다뤘다.
