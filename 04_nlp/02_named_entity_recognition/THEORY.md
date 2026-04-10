# 02 Named Entity Recognition 이론 노트

## 핵심 개념
- **named entity recognition (NER)** 은 문장 안에서 사람(PER), 기관(ORG), 장소(LOC) 같은 span을 찾아 token 단위 label로 표시하는 문제다.
- **BIO tagging** 은 entity의 시작을 `B-타입`, 내부를 `I-타입`, 나머지를 `O` 로 적는 가장 기본적인 span 표현 규약이다.
- **label alignment** 는 word-level gold label을 subword / word-piece token에 맞게 늘려 붙이는 과정이다. 첫 piece는 원래 label을 유지하고, 뒤 piece는 보통 `I-타입` 으로 변환한다.
- **token accuracy** 는 각 token label을 얼마나 맞혔는지 보지만, **entity-level precision / recall / F1** 은 span 경계를 올바르게 복원했는지 더 직접적으로 본다.
- **sequence labeler** 는 문장 전체 token 흐름을 보고 각 위치의 label을 예측한다. 같은 token이라도 앞뒤 문맥에 따라 다른 label을 받을 수 있다는 점이 중요하다.

## 왜 NER에서 alignment를 먼저 확인해야 하는가
NER를 처음 배울 때는 모델 아키텍처에만 시선을 빼앗기기 쉽다. 하지만 실제로는 다음 질문이 먼저다.

1. gold label이 단어 기준인지 subword 기준인지?
2. tokenizer가 단어를 몇 개 piece로 쪼개는지?
3. 쪼개진 뒤 `B-` 와 `I-` 를 어떻게 확장할지?
4. 평가를 token 기준으로 볼지 entity 기준으로 볼지?

이 네 질문을 놓치면 모델이 좋아도 결과 해석이 어긋난다.

## BIO 직관
예를 들어 `서울 시청` 이 하나의 장소 entity라면 가장 단순한 word-level tag는 `B-LOC`, `I-LOC` 이다. 그런데 tokenizer가 `시청` 을 `시`, `##청` 처럼 나누면 aligned tag는 보통 `B-LOC`, `I-LOC`, `I-LOC` 이 된다.

즉 entity는 "단어 수"가 아니라 **token span** 으로 다시 써야 한다. 이때 첫 조각은 시작(`B-`), 뒤 조각은 내부(`I-`)로 보는 습관이 중요하다.

## 왜 token accuracy만 보면 위험한가
NER는 `O` label 비중이 큰 경우가 많다. 그러면 모델이 entity 경계를 자주 틀려도 token accuracy는 꽤 높게 보일 수 있다. 하지만 실제 활용에서는 다음이 더 중요하다.

- 사람 이름 전체를 제대로 잡았는가?
- 기관 span을 중간에서 끊지 않았는가?
- `B-ORG` 와 `B-LOC` 를 헷갈리지 않았는가?

그래서 token accuracy와 함께 entity-level F1을 반드시 같이 읽어야 한다.

## tiny sequence labeler 직관
이 단위의 PyTorch demo는 다음처럼 아주 작은 구조를 쓴다.

- piece token -> token id
- token id -> embedding vector
- biGRU -> 각 token 위치의 contextual hidden state
- linear head -> 각 token의 BIO label logits

transformer만큼 강하지는 않지만, 적어도 **앞뒤 문맥을 보고 각 위치 label을 갱신하는 sequence labeling 감각**은 충분히 보여 준다.

## Common Confusion
- token accuracy가 높으면 entity extraction도 잘 된다고 생각하는 실수
- `B-` 와 `I-` 차이를 단순 formatting 문제로 여기는 실수
- alignment를 잘못했는데 모델 문제라고 착각하는 실수
- tiny toy dataset에서 나온 F1을 실제 benchmark 감각으로 일반화하는 실수

## 실행 결과 예시
이 단위에서는 실행 후 이런 식의 숫자를 읽게 된다.

```json
{
  "token_accuracy": 0.857143,
  "entity_precision": 0.8,
  "entity_recall": 0.8,
  "entity_f1": 0.8,
  "label_counts": {
    "O": 26,
    "B-PER": 5,
    "I-PER": 2,
    "B-ORG": 4,
    "I-ORG": 3,
    "B-LOC": 5,
    "I-LOC": 4
  }
}
```

```json
{
  "token_accuracy": 0.892857,
  "entity_f1": 0.857143,
  "embedding_dim": 20,
  "hidden_dim": 24,
  "epochs": 120,
  "num_labels": 7
}
```

숫자 자체보다 더 중요한 해석은 다음과 같다.
- scratch baseline이 특정 piece에서 강하게 맞히더라도, unseen boundary에서는 바로 깨질 수 있다.
- tiny neural model이 더 안정적이라면, 문맥 정보를 이용해 `이 token이 entity 안쪽인지 바깥인지` 를 더 잘 읽었을 가능성이 있다.
- token accuracy와 entity F1 차이가 크게 벌어지면, boundary error가 주요 실패 원인일 수 있다.

## 다음 단계로 이어지는 질문
- CRF decoding을 추가하면 boundary consistency가 얼마나 좋아질까?
- subword tokenizer 종류(BPE, WordPiece, unigram)에 따라 alignment 비용이 어떻게 달라질까?
- label imbalance가 큰 실제 NER 데이터셋에서 macro / micro entity F1을 어떻게 함께 볼까?
- pretrained encoder를 붙이면 unseen entity surface form을 얼마나 더 흡수할 수 있을까?
