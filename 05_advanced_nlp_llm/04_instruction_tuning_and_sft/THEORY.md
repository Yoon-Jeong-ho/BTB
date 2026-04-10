# 04 Instruction Tuning and SFT 이론 노트

## 핵심 개념

### 1. instruction tuning intuition: base LM을 assistant behavior로 다시 맞추기
- pretraining된 language model은 기본적으로 "다음 토큰 continuation"을 잘 맞히도록 학습되어 있다.
- 그런데 우리가 원하는 assistant behavior는 보통 다음 질문을 함께 포함한다.
  - 사용자의 요청을 어디까지 따라야 하는가?
  - 어떤 형식과 톤으로 답해야 하는가?
  - 모호한 지시를 어떤 방식으로 해석해야 하는가?
  - role/system constraint를 어떻게 반영해야 하는가?
- instruction tuning은 이런 사용성 문제를 해결하기 위해 **instruction-response 형식의 supervision** 을 추가하는 단계다.
- 직관적으로는 모델에게 "텍스트를 이어 쓰는 법"만이 아니라 **지시를 읽고 응답하는 패턴** 을 많이 보여 주는 과정이다.
- 보통 구현은 여전히 next-token loss를 쓰지만, 데이터가 plain corpus가 아니라 instruction dataset으로 바뀌면서 모델 행동이 assistant 쪽으로 이동한다.

### 2. supervised fine-tuning(SFT)은 무엇을 학습시키는가
- SFT는 labeled example `(prompt, ideal_response)` 또는 `(messages, assistant_reply)`를 이용해 **정답 응답을 모방** 하도록 학습한다.
- 흔한 데이터 모양은 다음과 같다.
  - instruction + optional context + response
  - system / user / assistant multi-turn conversation
  - task-specific input + reference answer
- 핵심은 모델이 "좋다/나쁘다"를 직접 배우는 것이 아니라, **주어진 정답 답변을 그대로 잘 생성하도록** gradient를 받는다는 점이다.
- 그래서 SFT는 아래에 강하다.
  - 원하는 출력 형식 맞추기
  - 질문-응답 인터페이스에 적응하기
  - tone, brevity, style, refusal phrasing 같은 표면 행동 정렬
- 반면 SFT만으로는 "여러 괜찮은 답 중 무엇이 더 선호되는가" 같은 preference ranking을 충분히 다루지 못한다.

### 3. input-output templating: 같은 예시도 포맷이 달라지면 학습 신호가 달라진다
- instruction tuning에서는 raw example 자체만큼이나 **어떻게 직렬화(serialization)하는가** 가 중요하다.
- 예를 들어 같은 요청이라도 다음과 같이 표현할 수 있다.
  - plain template
    - `### Instruction: ...`
    - `### Response: ...`
  - chat template
    - `<|system|> ...`
    - `<|user|> ...`
    - `<|assistant|> ...`
- template는 단순 꾸밈이 아니라 모델이 "이 토큰 이후에는 어떤 역할의 텍스트가 와야 하는가"를 구분하게 만드는 조건부 구조다.
- 그래서 template 설계는 다음 문제에 영향을 준다.
  - role boundary를 모델이 얼마나 안정적으로 구분하는가
  - generation 시작점이 어디인지
  - stop token / EOS 처리 방식이 어떤가
  - multi-turn history를 어떤 순서와 표식으로 연결하는가
- 잘못된 template는 모델이 prompt 일부를 다시 복창하거나, assistant boundary를 놓치거나, 불필요한 role tag를 출력하게 만들 수 있다.

### 4. role/system/user framing basics: 역할 구분은 conditioning signal이다
- chat-style SFT에서는 보통 `system`, `user`, `assistant` 세 역할을 구분한다.
- 매우 거칠게 보면 다음처럼 이해할 수 있다.
  - `system`: 전체 답변 정책, persona, 스타일, 제약
  - `user`: 현재 turn의 요청/질문
  - `assistant`: 모델이 생성해야 할 응답
- 중요한 점은 role tag가 단순한 주석이 아니라 **모델이 다음 토큰 분포를 바꾸는 입력 신호** 라는 것이다.
- 예를 들어 같은 user 질문이라도 system에 `간결하게 답하라`가 있으면 짧고 정리된 답을 내기 쉬워지고, `예시를 포함하라`가 있으면 구조가 달라질 수 있다.
- multi-turn에서는 이전 assistant 답변까지 함께 들어가므로, 모델은 대화 상태를 serialized history 속에서 추적하게 된다.
- 즉 role framing은 기억 모듈이 아니라 **입력으로 준 대화 이력과 역할 표식에 조건부로 반응하는 방식** 으로 작동한다.

### 5. SFT loss intuition: 보통은 assistant 구간을 더 중요하게 본다
- 구현 레벨에서는 chat template 전체를 하나의 토큰 시퀀스로 펼치고 next-token loss를 계산하는 경우가 많다.
- 하지만 실무/교육 관찰 포인트는 보통 다음 질문에 있다.
  - prompt/user/system 토큰에도 loss를 걸 것인가?
  - 아니면 assistant response 구간에만 loss를 집중할 것인가?
- assistant-only loss masking을 쓰면 모델은 "주어진 prompt를 다시 복사"하기보다 **응답 부분을 잘 생성하는 것** 에 더 집중한다.
- 반대로 전체 토큰에 loss를 걸면 template 복원 자체도 더 강하게 학습할 수 있지만, 학습 목표가 산만해질 수 있다.
- 여기서 핵심은 SFT가 전혀 새로운 objective라기보다, **serialized instruction example 위에서 next-token loss를 어디에 강조할지 조정하는 방식** 이라는 점이다.

### 6. helpfulness vs imitation: SFT가 잘하는 것과 못하는 것
- SFT는 helpfulness를 어느 정도 끌어올릴 수 있다. 이유는 training data 안에 이미 "좋은 assistant 답변처럼 보이는 형식"이 들어 있기 때문이다.
- 하지만 SFT의 본질은 여전히 imitation이므로 다음 한계가 있다.
  - 정답 하나를 과도하게 canonical answer처럼 외우기 쉽다.
  - 데이터에 자주 나온 style을 과도하게 복사할 수 있다.
  - 여러 가능한 답 중 더 선호되는 답을 분리해 내기 어렵다.
  - 장황함, 안전한 상투구, template overfitting이 생길 수 있다.
- 그래서 SFT 뒤 모델이 더 공손하고 그럴듯해 보여도, 그것이 곧 더 유용하거나 더 선호되는 답변을 안정적으로 고른다는 뜻은 아니다.
- 이 지점이 preference optimization이나 reward modeling이 뒤에 붙는 이유다.

### 7. SFT는 pretraining을 지우는가, 아니면 덮어쓰는가
- 보통 SFT는 base LM의 언어 능력을 완전히 새로 만드는 것이 아니라, 이미 학습된 지식을 **instruction interface 위로 다시 정렬** 한다.
- 그래서 data 규모와 learning rate가 너무 크면 base capabilities를 일부 망가뜨릴 수 있고, 너무 약하면 assistant behavior shift가 충분하지 않을 수 있다.
- 이 균형을 볼 때 자주 함께 등장하는 질문은 다음과 같다.
  - instruction data diversity가 충분한가?
  - formatting noise가 너무 크지 않은가?
  - style alignment만 생기고 reasoning quality는 그대로 아닌가?
  - domain-specific SFT가 일반 성능을 과하게 해치지 않는가?
- 즉 SFT는 "assistant화"의 첫 단계이지, 모든 능력 문제를 단번에 해결하는 마지막 단계는 아니다.

## Common Confusion
- SFT를 pretraining과 완전히 다른 objective라고 생각하는 실수
  - 실제로는 serialized instruction example 위에서 next-token loss를 쓰는 경우가 많다.
- role tag를 단순 메타데이터라고 생각하는 실수
  - system/user/assistant 표식은 실제 조건부 신호로 작동한다.
- instruction template만 맞추면 자동으로 더 좋은 답이 나온다고 생각하는 실수
  - format consistency와 true helpfulness는 다른 문제다.
- SFT가 preference learning까지 끝내 준다고 생각하는 실수
  - SFT는 정답 모방 중심이라 선호 비교를 직접 학습하지 않는다.
- assistant처럼 말하면 실제로 더 정확하다고 착각하는 실수
  - 톤 정렬과 factual correctness는 별개로 검증해야 한다.
- multi-turn history가 있으면 모델이 "상태를 기억한다"고 과장하는 실수
  - 실제로는 context window 안에 serialized history가 있을 때만 그 조건을 활용한다.

## 이 단위에서 무엇을 관찰할 것인가
- same task를 plain instruction template와 chat template로 바꾸면 모델 입력 직렬화는 어떻게 달라지는가?
- system prompt 유무가 응답 톤, 제약, refusal phrasing에 어떤 차이를 만드는가?
- loss mask를 assistant 구간 중심으로 둘 때와 전체 토큰에 둘 때 관찰 포인트는 무엇인가?
- SFT 데이터가 다양하지 않으면 어떤 imitation bias나 canned response 문제가 생기는가?
- helpfulness가 높아 보이는 응답과 실제로 더 선호되는 응답은 어디서 갈라지는가?
- 다음 preference optimization 단계로 넘어가기 전에, SFT만으로 남는 문제는 정확히 무엇인가?
