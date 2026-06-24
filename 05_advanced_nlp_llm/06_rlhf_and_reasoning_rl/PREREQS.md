# 06 RLHF and Reasoning RL 선행 개념

## 꼭 알고 오면 좋은 것
- supervised fine-tuning(SFT)이 초기 assistant policy를 만드는 단계라는 감각
- chosen/rejected pair와 preference optimization(DPO / ORPO / KTO)의 high-level 차이
- reward, policy, rollout, advantage, KL anchor, policy update 같은 RL 용어의 직관
- PPO가 여기서는 full implementation이 아니라 reward를 높이되 policy drift를 제한하는 PPO-family framing으로 쓰인다는 점
- verifier signal, judge score, factual accuracy, safety metric이 서로 다른 평가 축이라는 이해
- reasoning RL이 긴 답변 생성이 아니라 reward shaping으로 검증 가능한 문제 풀이 행동을 미는 과정이라는 이해

## 빠른 자기 점검
- reward model을 truth engine이 아니라 preference proxy로 설명할 수 있는가?
- offline preference optimization과 online RLHF rollout이 직접 보는 데이터 분포의 차이를 말할 수 있는가?
- PPO-family policy update에서 KL guardrail을 왜 같이 봐야 하는지 설명할 수 있는가?
- verifier와 judge가 각각 어떤 bias와 blind spot을 갖는지 예를 들 수 있는가?
- reward hacking, verbosity inflation, over-refusal이 왜 RLHF와 reasoning RL의 핵심 failure mode인지 말할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [RL Primer for RLHF](../../docs/05_rl_primer_for_rlhf.md) — reward/policy/rollout/advantage/KL/PPO 용어를 LLM post-training 맥락으로 먼저 정리한다.
- [04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — RLHF 이전 초기 policy를 만든다.
- [05_preference_optimization_dpo_orpo_kto](../05_preference_optimization_dpo_orpo_kto/README.md) — offline preference objective와 online RLHF loop를 비교한다.
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — training regression과 failure slice 관찰 감각을 복습한다.
