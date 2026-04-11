# 04 Instruction Tuning and SFT 이론 노트

## 핵심 개념

### 1. instruction format: continuation을 assistant interaction으로 바꾸는 프레임
- base language model은 기본적으로 다음 토큰 continuation을 맞히도록 pretraining된다.
- instruction format은 같은 텍스트를 `요청 → 응답` 구조로 다시 배치해, 모델이 “이제 사용자의 요구에 답해야 한다”는 조건을 보게 만든다.
- plain instruction-response format은 `### Instruction`, `### Response`처럼 boundary를 명시한다.
- chat template는 `system`, `user`, `assistant` role tag를 사용해 더 세밀한 대화 구조를 제공한다.
- 이 포맷은 단순한 주석이 아니라 input-output template 자체가 모델의 조건부 분포를 바꾸는 신호다.

### 2. supervised fine-tuning: 정답 응답을 모방하도록 하는 단계
- SFT(supervised fine-tuning)는 `(prompt, reference response)` 또는 `(messages, assistant answer)` 형태의 labeled example을 사용한다.
- 보통 구현은 serialized sequence 위에서 next-token loss를 계산하지만, instruction dataset이므로 target은 assistant response 쪽으로 해석한다.
- 교육적으로 중요한 질문은 “전체 토큰을 복원하는가?”가 아니라 “prompt/system/user 토큰을 무시하고 assistant 답변 토큰에 loss를 주는가?”이다.
- assistant-only loss mask를 쓰면 모델은 prompt를 그대로 복창하기보다 주어진 입력에 대한 응답을 생성하는 쪽에 집중한다.
- 그래서 SFT는 새로운 reasoning 원리를 자동으로 만들기보다, 이미 가진 언어 능력을 instruction-following 행동으로 재정렬한다.

### 3. input-output template와 serialization
- 같은 예시라도 serialization이 다르면 token boundary, EOS 위치, generation 시작점이 달라진다.
- 좋은 template는 다음을 명확히 한다.
  - system 제약은 어디에 놓이는가?
  - user request는 어디서 끝나는가?
  - assistant response는 어디서 시작하고 어디서 멈추는가?
  - multi-turn history를 어떤 순서로 이어 붙이는가?
- 나쁜 template는 모델이 role tag를 출력하거나, user 질문을 다시 복사하거나, response boundary를 놓치게 만들 수 있다.
- 따라서 instruction tuning에서는 데이터 내용뿐 아니라 input-output template 품질도 supervised signal의 일부다.

### 4. role framing: system/user/assistant는 conditioning signal이다
- `system`은 전체 답변 정책, persona, 길이, 안전 제약 같은 상위 조건을 담는다.
- `user`는 현재 turn의 요청과 추가 context를 담는다.
- `assistant`는 SFT에서 모델이 모방해야 하는 reference answer를 담는다.
- 이 세 역할은 데이터베이스 메타데이터가 아니라, 실제 token sequence에 포함되어 다음 token distribution을 바꾸는 conditioning signal이다.
- 같은 user 질문이라도 system에 “한국어로 간결하게 답하라”가 있으면 더 짧고 제약을 지킨 답변을 만들도록 학습 신호가 바뀐다.

### 5. imitation vs helpfulness tradeoff
- SFT는 reference answer를 잘 따라 하도록 만든다. 그래서 형식 준수, 말투, 기본 도움말 패턴에는 강하다.
- 하지만 SFT의 중심은 imitation이다. reference가 장황하거나 canned response를 많이 포함하면 모델도 그 습관을 배울 수 있다.
- helpfulness는 “정답을 복사했는가”보다 넓다. 모호한 지시를 명확히 하거나, 필요 없는 장황함을 줄이고, 사용자의 실제 목적에 맞게 답해야 한다.
- 여러 가능한 답변 중 어느 쪽이 더 선호되는지는 SFT 데이터 하나만으로 직접 배우기 어렵다.
- 그래서 SFT 뒤에는 DPO/ORPO/KTO 같은 preference optimization 또는 reward-based alignment가 이어질 수 있다.

### 6. SFT가 하는 일과 하지 않는 일
- 하는 일: assistant interface 적응, role boundary 학습, 출력 형식 정렬, 기본적인 instruction following 모방.
- 하지 않는 일: 사실성 자동 보장, preference ranking 직접 학습, 안전성 완성, domain knowledge 자동 추가.
- DAPT가 “무엇을 더 알게 만들 것인가”에 가깝다면, SFT는 “그 지식을 어떤 assistant behavior로 드러낼 것인가”에 가깝다.

## 실행 결과 예시
```text
$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py
{
  "template_views": {
    "plain_instruction": {"target_region": "assistant_response_only"},
    "chat_template": {"roles": ["system", "user", "assistant"]}
  }
}

$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
{
  "framework": "deterministic_numeric_sft",
  "loss_mask_summary": {
    "prompt_loss_tokens": 0,
    "assistant_loss_tokens": 58
  }
}
```

## 자주 헷갈리는 지점
- instruction format을 프롬프트 꾸밈이라고만 보는 실수: 실제로는 boundary와 target framing을 정한다.
- supervised fine-tuning을 pretraining과 완전히 다른 objective라고 보는 실수: toy 실습에서는 next-token loss 위에 assistant-only mask를 얹어 관찰한다.
- system/user/assistant role tag를 단순 메타데이터라고 보는 실수: role framing은 conditioning signal이다.
- SFT가 helpfulness를 완성한다고 보는 실수: SFT는 reference imitation에 강하고 preference 비교는 약하다.
- 말투가 assistant처럼 바뀌면 factual correctness도 좋아졌다고 착각하는 실수: 형식 정렬과 사실성 검증은 별도다.

## 이 단위에서 무엇을 관찰할 것인가
- plain instruction format과 chat template의 token boundary가 어떻게 다른가?
- assistant-only loss mask는 어떤 prompt/system/user 토큰을 무시하고 무엇을 target으로 남기는가?
- system message가 응답 톤과 제약을 바꾸는 signal로 어떻게 관측되는가?
- SFT training curve가 format imitation을 빠르게 높여도 helpfulness proxy가 같은 속도로 오르지 않는 이유는 무엇인가?
- preference optimization으로 넘어가기 전에 SFT만으로 남는 문제는 무엇인가?
