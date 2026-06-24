# 05 RL Primer for RLHF

이 문서는 `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`에 들어가기 전, 강화학습(RL) 용어를 LLM post-training 맥락으로 읽기 위한 짧은 primer다. BTB의 목적은 로봇 제어 전체 RL 교과서를 만드는 것이 아니라, RLHF와 reasoning RL 실험 로그를 읽을 수 있는 최소 공통 언어를 제공하는 것이다.

## 핵심 용어

- **policy**: 상태나 prompt를 보고 다음 행동/토큰을 선택하는 규칙이다. LLM에서는 policy model이 답변 token 분포를 만든다.
- **reward**: 행동이나 답변이 얼마나 좋은지 평가하는 숫자다. RLHF에서는 reward model, preference judge, rule-based verifier가 reward 역할을 할 수 있다.
- **rollout**: policy가 실제로 여러 step을 생성해 만든 trajectory다. LLM에서는 prompt에서 시작해 answer 또는 reasoning trace를 끝까지 생성한 결과다.
- **advantage**: 어떤 행동이 현재 기준선보다 얼마나 나았는지를 나타내는 신호다. 좋은 rollout을 더 밀고 나쁜 rollout을 덜 밀기 위해 쓴다.
- **KL anchor**: RL 업데이트가 base/SFT model에서 너무 멀어지지 않도록 잡아 주는 제약이다. 답변이 reward hack으로 무너지는 것을 줄인다.
- **PPO**: policy를 조금씩 업데이트하면서 reward를 올리고 KL drift를 제한하는 대표 online RL 알고리즘이다.

## RLHF에서의 최소 흐름

1. SFT model이 prompt에 대한 답변을 만든다.
2. reward model 또는 verifier가 답변을 점수화한다.
3. rollout 점수와 baseline을 비교해 advantage를 만든다.
4. policy를 reward가 높은 방향으로 업데이트한다.
5. KL penalty로 원래 모델에서 너무 멀어지는 것을 막는다.
6. win-rate, reward, KL, length, failure case를 함께 본다.

## Offline preference와 online RLHF의 차이

- **Offline preference optimization(DPO/ORPO/KTO)**: 이미 모아 둔 선호 쌍을 사용해 policy를 업데이트한다. rollout을 새로 많이 만들지 않아 상대적으로 단순하다.
- **Online RLHF/PPO**: 현재 policy가 새 답변을 생성하고, 그 결과를 reward로 평가해 다시 policy를 업데이트한다. reward hacking, KL collapse, sampling instability를 더 신경 써야 한다.

## 체크리스트

`05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`에 들어가기 전에 아래를 말로 설명한다.

- reward가 높은 답변이 항상 좋은 답변은 아닐 수 있는 이유
- rollout 길이가 reward와 KL을 동시에 흔드는 이유
- KL anchor를 빼면 policy가 어떤 식으로 망가질 수 있는지
- DPO와 PPO가 “선호를 반영한다”는 점은 같지만 데이터 생성 방식이 다른 이유

## 다음 연결

- [05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto](../05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/README.md)
- [05_advanced_nlp_llm/06_rlhf_and_reasoning_rl](../05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/README.md)
- [10_vla/01_vision_language_action_grounding](../10_vla/01_vision_language_action_grounding/README.md)
