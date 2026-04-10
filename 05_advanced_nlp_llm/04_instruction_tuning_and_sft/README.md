# 04 Instruction Tuning and SFT

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
base LM이 다음 토큰을 잘 맞힌다고 해서 곧바로 **사람이 원하는 방식으로 응답하는 assistant** 가 되지는 않는다. 실제 서비스형 모델은 보통 instruction 형식의 데이터로 다시 fine-tuning되며, 이때 모델은 단순 continuation보다 **질문-지시-응답의 역할 구조** 를 더 강하게 학습한다. 이 단위는 supervised fine-tuning(SFT)을 "성능 향상 마법"이 아니라 **입력 프레이밍과 정답 응답 패턴을 다시 맞추는 단계** 로 이해하게 만들어, 다음 단위의 preference optimization이 왜 추가로 필요한지도 준비시킨다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- instruction format, template, role framing, SFT trade-off를 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - prompt template별 input-output serialization 비교
  - user/assistant loss mask 관찰 메모
  - system prompt 유무에 따른 응답 변화 요약
  - helpfulness vs imitation trade-off 사례 정리

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. base pretraining 데이터와 instruction dataset을 구분하며, 같은 사실 질문도 왜 "문서 continuation"보다 "지시-응답" 형식으로 다시 보여 주는지 본다.
2. 단일 turn 예시를 `system`, `user`, `assistant` 역할이 있는 chat template와 plain instruction-response template 두 방식으로 serialization해 본다.
3. SFT에서 실제 loss가 어디에 걸리는지 본다. 보통 전체 시퀀스를 next-token 방식으로 학습하더라도, 관찰 포인트는 **assistant 답변 구간에 loss를 집중시키는가** 여부다.
4. system message를 넣었을 때 모델이 어떤 톤/제약을 따르기 쉬워지는지, role framing이 conditioning으로 어떻게 작동하는지 비교한다.
5. SFT가 helpfulness를 높이는 대신 단순 모방(imitation)이나 format overfitting을 만들 수 있는 지점을 따로 정리한다.
6. 마지막에는 "좋은 답변처럼 보이게 만드는 것"과 "사람이 더 선호하는 답변을 고르게 만드는 것"의 차이를 질문으로 남기며 `05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto`로 연결한다.

## 이 단위에서 특히 볼 질문
- instruction tuning은 pretraining objective를 완전히 바꾸는 것인가, 아니면 데이터 프레이밍과 supervision target을 재정렬하는 것인가?
- 같은 내용이라도 plain text continuation과 instruction-response template는 모델이 배우는 행동을 어떻게 다르게 만드는가?
- `system` / `user` / `assistant` 역할 구분은 단순 메타데이터인가, 아니면 실제 conditioning signal인가?
- SFT는 helpfulness를 높이면서도 왜 imitation bias나 canned response 문제를 함께 만들 수 있는가?
- assistant 구간에만 loss를 주는 설정과 전체 토큰에 loss를 주는 설정은 무엇을 다르게 학습시키는가?
- 왜 SFT만으로는 "더 선호되는 답변" 선택 문제가 충분히 해결되지 않고, 다음 단위의 preference optimization이 이어지는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/scratch_lab.py
{
  "status": "sample",
  "template_views": [
    {
      "name": "plain_instruction",
      "serialized_prefix": "### Instruction:\n한국어로 요약하라",
      "target_region": "assistant_response_only",
      "loss_tokens": 42
    },
    {
      "name": "chat_template",
      "roles": ["system", "user", "assistant"],
      "serialized_prefix": "<|system|>친절하고 간결하게 답하라",
      "target_region": "assistant_response_only",
      "loss_tokens": 39
    }
  ],
  "framing_observation": {
    "with_system_message": "tone more constrained",
    "without_system_message": "answer less policy-shaped"
  }
}

$ python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
{
  "status": "sample",
  "batch_shape": {
    "input_ids": [4, 512],
    "labels": [4, 512],
    "loss_mask": [4, 512]
  },
  "masked_regions": {
    "prompt_tokens": 0,
    "assistant_tokens": 133
  },
  "notes": [
    "SFT still uses next-token loss on serialized conversations",
    "helpfulness gain does not guarantee preference alignment"
  ]
}
```

핵심은 숫자 자체보다도 **template가 어떻게 직렬화되는지**, **loss가 어느 역할 구간에 걸리는지**, **system/user framing이 응답 톤과 제약에 어떤 조건부 신호를 주는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 SFT를 "instruction 형식에 맞춘 supervised imitation"으로 정리해 두면, 다음 단위 `05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto`에서 왜 단순 정답 모방을 넘어 **두 응답 중 어느 쪽을 더 선호하는가** 를 학습해야 하는지 자연스럽게 이어진다. 즉, SFT는 assistant의 기본 말투와 형식을 맞추는 첫 단계이고, preference optimization은 그 위에서 **더 나은 응답 선택 기준** 을 추가로 밀어 넣는 단계다.
