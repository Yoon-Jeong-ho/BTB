# 03 Machine Reading Comprehension 이론 노트

## 핵심 개념
- **machine reading comprehension (MRC)** 는 질문과 문맥을 함께 읽고 문맥 안 span을 정답으로 고르거나, 정답이 없다고 판단하는 문제다.
- **span extraction** 은 context token sequence 안에서 `(start, end)` 위치를 예측하는 방식이다.
- **exact match (EM)** 는 예측 문자열이 gold answer와 완전히 같을 때만 1이 된다.
- **token F1** 은 예측 span과 gold span이 일부만 겹쳐도 부분 점수를 줄 수 있어서 boundary error를 더 세밀하게 보여 준다.
- **answerable / unanswerable 판단** 은 span 점수만큼 중요하다. 질문에 답이 없는데도 아무 span이나 고르면 QA 모델은 실제로는 신뢰하기 어렵다.
- **no-answer threshold** 는 "이 점수보다 낮으면 차라리 답하지 말자" 를 정하는 기준이다.

## 왜 첫 독해 실습에서 threshold가 중요한가
독해 문제를 처음 배우면 정답 span만 잘 고르면 된다고 느끼기 쉽다. 하지만 실제 QA 시스템에서는 다음 질문이 먼저다.

1. 질문과 가장 관련 있는 문맥 window가 어디인가?
2. 그 window 안에서 어느 span이 정답처럼 보이는가?
3. span 후보 점수가 충분히 높은가, 아니면 답하지 않는 편이 안전한가?
4. 모델이 틀렸다면 boundary를 틀린 것인가, 질문을 오독한 것인가, no-answer threshold가 잘못된 것인가?

이 단위는 바로 이 네 질문을 가장 작은 한국어 예제로 드러낸다.

## heuristic span extraction 직관
아주 작은 baseline이라도 다음 두 축을 분리해서 보면 독해가 덜 추상적으로 느껴진다.
- **question-context overlap**: 질문 핵심 token이 context 어느 window에서 다시 나타나는가?
- **answer type hint**: `어디`, `언제`, `누가` 같은 질문 형태에 맞는 span 길이와 token 패턴이 있는가?

문맥 window overlap이 충분하지 않으면 span 후보를 만들지 않고 no-answer로 보내는 것이 오히려 더 좋은 baseline이 될 수 있다.

## tiny PyTorch QA model 직관
이 단위의 framework 실험은 transformer 대신 아주 작은 QA-style 모델을 쓴다.

- `[CLS] question [SEP] context [SEP]` 형태로 시퀀스를 만든다.
- token embedding과 segment embedding을 합친 뒤 작은 biGRU encoder를 통과시킨다.
- encoder 출력으로 start / end logits를 뽑아 span을 찾는다.
- 동시에 `[CLS]` 쪽 representation으로 answerable 여부를 예측한다.

이 구조는 매우 작지만, 그래도 질문 표현을 context token마다 다시 섞어 보면서 span을 고른다는 점에서 heuristic baseline보다 한 단계 더 QA답다.

## EM과 token F1을 같이 보는 이유
- EM만 보면 정답을 거의 맞혔는데 조사 하나나 boundary 하나가 어긋난 경우도 0점이 된다.
- token F1은 일부 겹침을 반영하므로 partial span error를 더 빨리 보여 준다.
- 반대로 F1만 보면 answerable / unanswerable를 잘못 처리한 위험이 가려질 수 있다.
- 그래서 **EM + token F1 + answerable accuracy** 를 같이 봐야 span 품질과 abstention 품질을 함께 읽을 수 있다.

## Common Confusion
- EM이 낮으면 모델이 완전히 쓸모없다고 단정하는 실수
- token F1이 높으니 no-answer 처리도 잘 됐다고 착각하는 실수
- 질문 token overlap이 높으면 언제나 정답 span을 찾을 수 있다고 믿는 실수
- toy MRC에서 나온 threshold를 실제 benchmark에 그대로 일반화하는 실수

## 실행 결과 예시
이 단위에서는 실행 후 이런 식의 숫자를 읽게 된다.

```json
{
  "eval_exact_match": 0.5,
  "eval_token_f1": 0.866667,
  "answerable_accuracy": 1.0,
  "no_answer_threshold": 4.4175,
  "answerable_exact_match": 0.333333,
  "unanswerable_exact_match": 1.0
}
```

```json
{
  "eval_exact_match": 0.5,
  "eval_token_f1": 0.783333,
  "embedding_dim": 28,
  "hidden_dim": 24,
  "epochs": 160,
  "answerable_accuracy": 1.0
}
```

숫자 자체보다 중요한 것은 다음 해석이다.
- heuristic baseline의 EM이 높다면 question-context lexical alignment만으로도 풀리는 패턴이 많다는 뜻이다.
- token F1이 EM보다 더 높다면 boundary는 조금 흔들렸지만 정답 핵심 단어는 맞췄을 가능성이 있다.
- answerable accuracy가 낮다면 span extraction보다 먼저 no-answer threshold나 answerability head를 다시 봐야 한다.
- framework 모델이 baseline보다 좋아졌다면, 질문 summary를 context token마다 다시 조건부로 읽은 효과를 의심해 볼 수 있다.

## 다음 단계로 이어지는 질문
- 긴 문맥에서 retrieval 단계가 틀리면 span head는 얼마나 무력해질까?
- answerable / unanswerable 비율이 바뀌면 no-answer threshold는 어떻게 다시 잡아야 할까?
- pretrained encoder를 쓰면 tiny QA model 대비 어떤 표현 이득이 생길까?
- 오답을 읽을 때 boundary error와 질문 오독을 어떻게 구분할까?
