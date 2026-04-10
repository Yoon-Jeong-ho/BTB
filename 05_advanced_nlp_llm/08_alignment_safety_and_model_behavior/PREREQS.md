# 08 Alignment, Safety, and Model Behavior 선행 개념

## 꼭 알고 오면 좋은 것
- instruction tuning, preference optimization, RLHF가 모두 모델 행동을 바꾸는 post-training 단계라는 큰 그림
- prompt / response / role framing이 모델 행동에 조건부 신호를 준다는 점
- evaluation이 단일 점수보다 slice와 failure mode 관찰로 더 잘 읽힌다는 감각
- harmlessness, refusal, robustness가 서로 겹치지만 완전히 같은 말은 아니라는 점
- 모델 정책과 시스템 guardrail이 서로 다른 책임을 갖는다는 기본 이해
- "정확한 답을 낼 수 있음" 과 "배포 환경에서 바람직하게 행동함" 을 구분해야 한다는 점

## 빠른 자기 점검
- capability와 alignment를 한두 문장으로 분리해 설명할 수 있는가?
- refusal이 필요한 경우와 over-refusal이 되는 경우를 예로 들 수 있는가?
- harmlessness가 단순 무응답이 아니라 안전한 대안 제시까지 포함할 수 있다는 점을 받아들일 수 있는가?
- paraphrase나 formatting noise가 robustness 평가 대상인 이유를 말할 수 있는가?
- tool gating, access control, moderation 같은 시스템 장치가 왜 모델 자체와 별도로 필요한지 설명할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — role framing과 assistant behavior shaping의 출발점을 다시 본다.
- [05_preference_optimization_dpo_orpo_kto](../05_preference_optimization_dpo_orpo_kto/README.md) — 어떤 응답을 더 선호하거나 덜 선호하게 만들 것인지의 objective 감각을 복습한다.
- [06_rlhf_and_reasoning_rl](../06_rlhf_and_reasoning_rl/README.md) — reward/policy update 관점에서 behavior shaping이 어떻게 이어지는지 연결한다.
- [07_retrieval_augmented_generation_and_eval](../07_retrieval_augmented_generation_and_eval/README.md) — eval slice와 failure analysis를 시스템 관점으로 다시 묶는다.
