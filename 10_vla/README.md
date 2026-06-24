# 10 VLA

이 트랙은 `09_multimodal` 다음에 놓이는 **Vision-Language-Action(VLA) 연결 구간**이다. 이미지와 텍스트를 함께 이해하는 수준을 넘어, 관측 상태와 언어 지시를 action token, safety gate, trajectory/eval 관점으로 바꾸는 첫 단계를 다룬다.

## 선행 권장

- `00_foundations`의 tensor/gradient/runtime 감각
- `02_deep_learning/04_attention_and_transformers`
- `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`의 reward/policy 용어
- `08_multimodal_bridge -> 09_multimodal`의 image-text alignment, captioning, VQA 감각

## 단위 구성

| Unit | Status | Focus |
| --- | --- | --- |
| [01_vision_language_action_grounding](01_vision_language_action_grounding/README.md) | runnable | 시각 상태와 언어 지시를 action token / safety gate로 연결하는 최소 VLA grounding 실험 |

## 이 트랙에서 보는 질문

- 이미지/상태 관측과 자연어 지시가 action 선택으로 바뀌려면 어떤 중간 표현이 필요한가?
- VQA의 “답변”과 VLA의 “행동”은 평가 지표가 어떻게 다른가?
- action accuracy만 높아도 safety gate가 틀리면 왜 실패인가?
- 이후 behavior cloning, trajectory error, intervention count, safety violation 같은 지표로 어떻게 확장할 수 있는가?
