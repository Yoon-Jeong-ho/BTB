# 03 Machine Reading Comprehension

## 왜 이 단위를 배우는가
`03_nlp/02_named_entity_recognition`에서 span 경계를 토큰 단위로 다뤄 봤다면, 이제는 **질문과 문맥을 함께 읽고 정답 span을 고르거나 아예 답하지 않는 판단**까지 해 보는 applied NLP 단계가 필요하다. 이 단위는 작은 한국어 예제로 `question-context 정렬 -> heuristic span extraction -> tiny PyTorch QA head -> 분석 보고서` 흐름을 직접 돌리게 해서, 이후 KLUE-MRC나 SQuAD 2.0으로 넘어갈 때도 **정답 span과 no-answer threshold를 먼저 해석하는 습관**을 만들도록 설계했다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/answerability_breakdown.svg`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 실행별 관측치를 적는 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 프레임을 담은 `analysis.md`
- 학습자가 직접 적는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 toy 한국어 독해 예제를 이용해 question keyword와 local window overlap으로 span extraction baseline을 만든다.
2. 같은 예제를 `framework_lab.py`에서 tiny PyTorch QA-style model로 다시 학습해 start/end span과 answerable 여부를 함께 예측한다.
3. `analysis.py`로 exact match, token F1, no-answer threshold가 각각 무엇을 말해 주는지 한국어 문장으로 다시 정리한다.

## 이 단위에서 특히 볼 질문
- 질문 token과 문맥 token이 어디서 만나야 정답 span 후보가 살아나는가?
- 왜 exact match와 token F1을 함께 봐야 partial span error를 놓치지 않을까?
- unanswerable 질문에서 "아무 답이나 찍는 모델"과 "조심스럽게 abstain 하는 모델"을 어떻게 구분할까?
- tiny PyTorch QA model이 heuristic baseline보다 좋아 보인다면, 그 차이는 질문-문맥 상호작용 때문인가 threshold 운이 좋았기 때문인가?

## 실행 방법
```bash
python 03_nlp/03_machine_reading_comprehension/scratch_lab.py
python 03_nlp/03_machine_reading_comprehension/framework_lab.py
python 03_nlp/03_machine_reading_comprehension/analysis.py
```

## 실행 결과 예시
아래 숫자는 이 toy MRC unit에서 기대하는 출력 형식의 예시다. 실제 값은 seed나 학습 경로에 따라 조금 달라질 수 있지만, **metrics.json + svg + analysis report** 조합은 유지된다.

```json
{
  "train_size": 8,
  "eval_size": 4,
  "eval_exact_match": 0.5,
  "eval_token_f1": 0.866667,
  "answerable_accuracy": 1.0,
  "no_answer_threshold": 4.4175,
  "figure_path": "artifacts/scratch-manual/answerability_breakdown.svg"
}
```

```json
{
  "train_size": 8,
  "eval_size": 4,
  "embedding_dim": 28,
  "hidden_dim": 24,
  "eval_exact_match": 0.5,
  "eval_token_f1": 0.783333,
  "answerable_accuracy": 1.0
}
```

## 무엇을 읽고 다음 단계로 넘어가면 좋은가
1. [PREREQS.md](./PREREQS.md) — exact match, token F1, no-answer 판단이 왜 필요한지 빠르게 점검한다.
2. [THEORY.md](./THEORY.md) — span extraction과 abstention을 읽는 최소 개념 세트를 먼저 읽는다.
3. `scratch_lab.py` 출력과 `answerability_breakdown.svg` — heuristic baseline이 어떤 span을 고르고 어디서 멈추는지 확인한다.
4. `framework_lab.py` 출력 — 같은 문제를 tiny PyTorch QA model이 어떻게 다시 읽는지 확인한다.
5. `analysis.py`와 `analysis.md` — 숫자를 span 해석 문장으로 바꾸는 연습을 한다.

## 다음 단위와의 연결
이 감각이 있으면 KLUE-MRC, SQuAD 2.0, 또는 retrieval-augmented QA 실습으로 넘어갈 때도 먼저 **질문-문맥 정렬 sanity check -> EM/F1 + answerability 비교 -> failure span 해석** 순서를 밟게 된다. 그 습관이 있어야 pretrained QA model도 덜 블랙박스처럼 보인다.
