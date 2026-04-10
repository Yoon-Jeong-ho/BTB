# 02 Corpus, Tokenizer, and Data Mixture 이론 노트

## 핵심 아이디어
LLM의 pretraining signal은 objective만이 아니라 corpus와 tokenizer, 그리고 mixture sampling이 함께 만든다. 이 단위는 실제 대규모 학습 대신 작은 toy corpus로 **corpus quality**, **dedup / contamination**, **multilingual tokenizer tradeoff**, **domain balance**, **token budget**을 관찰한다.

## 1. corpus quality는 크기보다 유효 신호다
문서가 많아도 spam, boilerplate, exact duplicate, near duplicate가 많으면 모델은 같은 문장을 반복해서 보거나 노이즈를 학습한다. 반대로 너무 작고 깨끗한 corpus만 쓰면 coverage가 좁다. 좋은 corpus 설계는 raw size와 effective signal을 구분한다.

## 2. dedup과 contamination은 목적이 다르다
- **dedup**: train corpus 내부에서 같은 문서나 거의 같은 문서가 반복 노출되는 것을 줄인다.
- **contamination**: benchmark, validation, held-out test 문서가 train data에 섞여 평가가 부풀려지는 것을 막는다.
둘은 같은 문자열 검색으로 일부 겹칠 수 있지만, 관찰 질문이 다르다. 이 실습에서는 exact duplicate 1개, near duplicate 1개, contamination hit 2개를 분리해서 센다.

## 3. tokenizer tradeoff는 compression과 fairness를 동시에 바꾼다
`toy_unigram_like`는 긴 조각을 유지해 평균 token 수를 줄인다. `toy_aggressive_subword`는 한국어·일본어·영어 조각을 더 잘게 나눠 unknown 위험을 낮추는 대신 sequence length를 늘린다. multilingual shared tokenizer에서는 특정 언어가 더 잘게 쪼개져 같은 의미를 표현하는 데 더 많은 token budget을 쓸 수 있다.

## 4. mixture는 문서 수보다 token budget 기준으로 읽어야 한다
도메인별 문서 수가 같아도 문서 길이와 tokenizer가 다르면 실제 학습에서 쓰는 token share가 달라진다. 그래서 domain balance와 multilingual mixture는 sample count보다 token count로 다시 계산해야 한다.

## 실행 결과 예시
```text
scratch: raw=11, trainable=7, dedup_removed=2, contamination_blocked=2
framework: device=cpu, tokenizer=toy_unigram_like, context_window=64
analysis: artifacts/analysis-manual/latest_report.md 갱신
```

## Common Confusion
- corpus가 커지면 품질 문제가 자동으로 해결된다고 생각하는 것
- dedup과 contamination check를 같은 단계로 취급하는 것
- tokenizer를 compression 하나로만 평가하고 multilingual fragmentation을 놓치는 것
- mixture 비율을 문서 수로만 보고 token budget 차이를 보지 않는 것

## 이 단위에서 확인할 질문
- raw corpus에서 trainable corpus로 줄어드는 이유가 무엇인가?
- aggressive tokenizer가 sequence length와 context window 사용량을 얼마나 늘리는가?
- contamination을 제거하지 않으면 어떤 evaluation claim이 위험해지는가?
- domain balance를 token share로 다시 계산하면 어느 slice가 커지는가?
- multilingual mixture에서 한국어/영어/일본어 fragmentation 차이를 어떻게 읽을 수 있는가?
