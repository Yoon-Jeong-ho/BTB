# 01 Text Classification 이론 노트

## 핵심 개념
- **text classification**은 입력 문장을 미리 정해진 라벨(예: 긍정/부정, 주제 A/B/C) 중 하나로 보내는 문제다.
- **bag-of-words**는 단어의 순서를 버리고, 어떤 token이 몇 번 등장했는지만 세는 가장 단순한 표현 방식이다.
- **multinomial Naive Bayes**는 클래스마다 token이 얼마나 자주 나오는지 세어, 새 문장이 어느 클래스 vocabulary와 더 닮았는지 확률적으로 비교한다.
- **embedding-average classifier**는 token id를 dense vector로 바꾼 뒤 평균을 내고 linear head로 라벨을 예측하는 작은 neural baseline이다.
- **accuracy**는 전체 정답률, **macro F1**은 각 클래스 F1을 동등 가중 평균한 값이라서 클래스 불균형이나 특정 클래스 collapse를 더 잘 드러낸다.

## 왜 첫 applied NLP에서 baseline이 중요한가
텍스트 분류를 처음 배우면 곧바로 BERT fine-tuning부터 시작하고 싶어지기 쉽다. 하지만 실제로는 다음 질문이 먼저다.

1. 문장을 어떤 단위(token)로 자를 것인가?
2. 그 token을 어떤 숫자 표현으로 바꿀 것인가?
3. baseline이 이미 맞히는 패턴은 무엇인가?
4. neural model이 baseline보다 좋아졌다면, 무엇이 더 표현되었기 때문인가?

이 단위는 바로 이 네 질문을 제일 작은 데이터와 코드로 드러낸다.

## bag-of-words 직관
문장 순서를 완전히 무시하면 많은 정보가 사라지지만, 동시에 다음 같은 장점이 생긴다.
- 구현이 매우 단순하다.
- 어떤 token이 어느 클래스 쪽으로 기울었는지 설명하기 쉽다.
- baseline으로 삼아 이후 모델이 **정말로 더 좋은 표현을 배웠는지** 비교하기 좋다.

예를 들어 긍정 문장에 `친절`, `추천`, `만족`이 반복되고 부정 문장에 `느리다`, `오류`, `실망`이 반복되면, count만 세도 꽤 많은 예제가 풀린다.

## neural classifier 직관
PyTorch tiny classifier는 보통 다음 흐름을 가진다.

- token -> token id
- token id -> embedding vector
- 문장 전체 vector = token embedding 평균
- 문장 vector -> linear head -> logits

이 구조는 transformer보다 훨씬 작지만, 그래도 **dense representation**을 학습한다는 점에서 bag-of-words와 다르다. 비슷한 단어들이 비슷한 embedding 쪽으로 이동하면, training 데이터에 직접 없던 조합도 어느 정도 일반화할 수 있다.

## accuracy와 macro F1을 같이 보는 이유
- accuracy만 보면 다수 클래스를 계속 찍는 모델이 좋아 보일 수 있다.
- macro F1은 각 클래스를 같은 비중으로 보므로, 한쪽 클래스를 계속 놓치는 모델을 더 빨리 드러낸다.
- 첫 applied NLP 단계에서는 "정답률이 높다"보다 **어떤 문장을 어떤 근거로 맞히고 틀리는가**가 더 중요하다.

## Common Confusion
- bag-of-words가 단순하다고 해서 쓸모없다고 생각하는 실수
- neural model 점수가 조금 높으면 바로 "의미를 이해했다"고 과대해석하는 실수
- accuracy와 macro F1이 같은 질문에 답한다고 생각하는 실수
- toy dataset에서 나온 성능을 실제 대규모 benchmark 감각으로 일반화하는 실수

## 실행 결과 예시
이 단위에서는 실행 후 이런 식의 숫자를 읽게 된다.

```json
{
  "eval_accuracy": 0.75,
  "eval_macro_f1": 0.733333,
  "top_positive_tokens": ["친절", "추천", "만족"],
  "top_negative_tokens": ["실망", "오류", "느리다"]
}
```

```json
{
  "eval_accuracy": 0.75,
  "eval_macro_f1": 0.733333,
  "embedding_dim": 12,
  "epochs": 80,
  "label_names": ["negative", "positive"]
}
```

숫자 자체보다 중요한 것은 다음 해석이다.
- baseline의 top token signal이 분명하면, dataset 안에 표면 lexical cue가 많다는 뜻이다.
- neural model이 조금 더 안정적이면, 평균 embedding 표현이 일부 동의어/조합 변형을 흡수했을 가능성이 있다.
- 두 모델이 모두 틀리는 문장은 token count만으로는 부족하거나, 데이터가 너무 작아 일반화가 어려운 경우다.

## 다음 단계로 이어지는 질문
- TF-IDF를 넣으면 toy baseline이 어떻게 달라질까?
- 문장 길이 정보나 bigram을 추가하면 무엇이 개선될까?
- pretrained tokenizer와 encoder를 쓰면 어떤 정보가 새로 들어오는가?
- 오분류 분석을 통해 다음 실험 가설을 어떻게 만들까?
