# 01 Language Modeling and Pretraining Objectives 이론 노트

## 핵심 개념

### 1. prediction target framing: 무엇을 맞히게 할 것인가
- pretraining objective는 결국 **모델에게 어떤 입력을 보여 주고, 무엇을 정답으로 맞히게 할지** 를 정하는 규칙이다.
- 이 선택이 달라지면 다음 요소가 함께 바뀐다.
  - 어떤 문맥을 볼 수 있는가
  - 어떤 위치에 loss가 걸리는가
  - 정답이 token 단위인지, span 단위인지
  - pretraining 뒤에 얻기 쉬운 능력이 생성 쪽인지, representation 쪽인지, seq2seq 복원 쪽인지
- 같은 transformer 계열 모델이라도 objective가 다르면 학습 신호의 모양이 크게 달라진다. 그래서 objective는 단순한 구현 옵션이 아니라 **모델 행동을 미리 비틀어 두는 설계 선택** 에 가깝다.

### 2. causal language modeling: 왼쪽 문맥으로 다음 토큰 맞히기
- causal LM은 autoregressive setup의 가장 기본형이다.
- 입력이 `[x_1, x_2, ..., x_t]` 라면, 모델은 보통 각 시점에서 **다음 토큰** 을 맞히도록 학습된다.
  - 입력: `[BOS, 나는, 오늘, 학교에]`
  - 정답: `[나는, 오늘, 학교에, 갔다]`
- 그래서 실습에서는 대개 입력과 타깃을 한 칸 shift해 놓고 본다.
- 핵심 특징은 다음과 같다.
  - 각 위치는 **왼쪽(prefix) 문맥만** 볼 수 있다.
  - loss는 보통 대부분의 시점에 걸린다.
  - generation 시 추론 방식과 학습 방식이 비교적 자연스럽게 이어진다.
- 직관적으로는 "앞에서부터 읽으며 다음 한 칸을 계속 예측하는 학습" 이다. 그래서 텍스트 생성과 잘 맞지만, 현재 위치의 오른쪽 문맥을 직접 볼 수는 없다.

### 3. masked language modeling: 가린 위치만 복원하기
- masked LM은 입력 일부를 가리고, 그 가려진 위치의 원래 token을 맞히게 만든다.
- 예를 들어
  - 원문: `고양이는 창문 위에 앉아 있다`
  - 입력: `고양이는 [MASK] 위에 앉아 있다`
  - 정답: `창문`
- 핵심 특징은 다음과 같다.
  - 가려진 위치를 예측할 때 **양쪽 문맥** 을 모두 활용할 수 있다.
  - loss는 전체 토큰이 아니라 **mask된 위치에만** 걸린다.
  - representation learning이나 bidirectional encoder pretraining과 잘 맞는다.
- 여기서 중요한 점은 "양방향 문맥을 본다"가 곧 치팅이라는 뜻이 아니라는 것이다. 모델은 정답 token 자체를 입력으로 보지 못하고, **빈칸 주변 단서** 로 맞혀야 한다.
- 다만 pretraining 때 `[MASK]` 같은 인위적 corruption을 쓰기 때문에, downstream 사용 시 입력 분포와 완전히 같지는 않다는 점도 함께 봐야 한다.

### 4. span corruption: 여러 토큰 덩어리를 통째로 복원하기
- span corruption은 token 하나가 아니라 **연속된 span** 을 지우고 복원하게 만드는 objective다.
- 예를 들어
  - 원문: `고양이는 창문 위에 앉아 있다`
  - encoder 입력: `고양이는 <extra_id_0> 앉아 있다`
  - decoder 정답: `<extra_id_0> 창문 위에`
- 보통 T5 스타일 설명에서는 여러 span을 sentinel token(`<extra_id_0>`, `<extra_id_1>` 등)으로 치환하고, decoder가 빠진 조각들을 순서대로 복원한다.
- 핵심 특징은 다음과 같다.
  - corruption 단위가 token 하나가 아니라 **span** 이다.
  - encoder-decoder setup과 자연스럽게 연결된다.
  - 입력 문서의 일부를 압축해서 보고, decoder에서 빠진 조각을 복원하는 denoising pretraining 성격이 강하다.
- masked LM보다 더 큰 조각을 복원하므로 "문맥 사이의 연결 구조를 복원하는 힘" 을 보기 좋고, 생성형 decoder target도 함께 다루게 된다.

### 5. context window intuition: 창문 크기와 예측 규칙은 같은 말이 아니다
- context window는 한 번의 forward에서 모델이 참고할 수 있는 token budget을 뜻한다.
- 하지만 이것만으로 objective의 성격이 자동 결정되지는 않는다.
  - causal LM: 각 위치는 window 안의 **왼쪽 토큰만** 본다.
  - masked LM: mask된 위치는 window 안의 **양쪽 토큰** 을 참고한다.
  - span corruption: encoder는 corruption된 문서를 보고, decoder는 sentinel을 기준으로 빠진 span을 생성한다.
- 즉 window는 "얼마나 멀리까지 볼 수 있는가" 에 대한 제약이고, objective는 "그중 무엇을 보게 하고 무엇을 맞히게 하는가" 에 대한 규칙이다.
- 또 context window는 장기 기억 그 자체가 아니다. 긴 window가 있다고 해서 먼 정보를 자동으로 잘 활용하는 것도 아니고, 짧은 window라고 해서 반드시 local pattern만 배우는 것도 아니다. **무엇을 예측하도록 학습했는가** 가 함께 중요하다.

### 6. prediction target framing이 downstream 감각을 어떻게 바꾸는가
- causal LM은 next-token continuation과 generation 흐름을 직접 학습하므로, text completion / assistant generation 같은 later task와 연결이 쉽다.
- masked LM은 빈칸 복원과 bidirectional representation에 강점이 있어 classification, tagging, retrieval encoder처럼 "입력을 잘 읽는" 쪽 감각과 잘 맞는다.
- span corruption은 encoder-decoder 기반 summarization, translation, instruction-style text-to-text framing으로 이어지기 좋다.
- 물론 실제 능력은 architecture, data mixture, scale, training recipe에도 좌우되지만, objective는 그 능력의 **초기 방향성** 을 정한다.

## 자주 헷갈리는 지점
- objective와 architecture를 같은 것으로 보는 실수
  - transformer는 causal LM, masked LM, span corruption 모두에 쓰일 수 있다.
- masked LM이 곧 생성 모델이라고 생각하는 실수
  - 빈칸 복원과 free-form autoregressive generation은 같은 문제가 아니다.
- causal LM은 "문장 전체를 한 번에 맞힌다"고 이해하는 실수
  - 실제로는 위치별 next-token prediction loss가 누적된다.
- span corruption을 "mask를 더 많이 친 것" 정도로만 이해하는 실수
  - token 단위 예측이 아니라 span 단위 복원 + sentinel bookkeeping이 핵심이다.
- context window를 곧 장기 기억이라고 착각하는 실수
  - window는 볼 수 있는 범위 제약이고, 실제 활용 여부는 objective와 optimization에 달려 있다.
- loss가 많이 걸리면 항상 더 좋은 objective라고 단정하는 실수
  - supervision density, corruption difficulty, downstream alignment를 같이 봐야 한다.

## 이 단위에서 무엇을 관찰할 것인가
- 같은 문장을 세 objective로 바꿨을 때 input/target pair가 어떻게 달라지는가?
- loss가 걸리는 위치와 개수는 objective마다 얼마나 다른가?
- causal LM의 "왼쪽만 보기" 와 masked LM의 "양쪽 보기" 가 실제 예측 난이도를 어떻게 바꾸는가?
- span corruption에서 sentinel token이 왜 필요한가, 그리고 decoder target 길이는 어떻게 달라지는가?
- context window를 늘렸을 때 objective별로 실제로 도움이 되는 관찰 포인트는 무엇인가?
- later tokenizer/data mixture/unit planning에서 objective 차이를 어떤 질문으로 다시 가져갈 수 있는가?
