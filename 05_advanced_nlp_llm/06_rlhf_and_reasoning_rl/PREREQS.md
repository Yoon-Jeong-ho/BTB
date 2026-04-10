# 06 RLHF and Reasoning RL 선행 개념

## 꼭 알고 오면 좋은 것
- supervised fine-tuning(SFT)이 초기 assistant policy를 만드는 단계라는 기본 감각
- chosen / rejected pair와 preference optimization(DPO / ORPO / KTO)의 high-level 차이
- reward, policy, rollout, update 같은 RL 용어를 아주 거칠게라도 읽을 수 있는 정도의 감각
- offline dataset 기반 최적화와 online rollout 기반 최적화가 왜 다른지에 대한 직관
- held-out evaluation, regression check, distribution shift의 기본 개념
- judge score, verifier signal, factual accuracy, safety metric이 서로 다른 평가 축이라는 이해

## 빠른 자기 점검
- reward model을 "절대 정답 판별기" 가 아니라 "선호 proxy" 로 설명할 수 있는가?
- offline preference optimization과 online RLHF가 각각 어떤 데이터를 직접 보고 policy를 바꾸는지 구분할 수 있는가?
- reasoning 품질을 높인다는 말이 단순히 더 긴 답변을 만드는 것과 같지 않다는 점을 받아들일 수 있는가?
- verifier와 judge가 서로 다른 종류의 편향과 blind spot을 가진 신호원이라는 점을 설명할 수 있는가?
- reward hacking, verbosity inflation, over-refusal 같은 regression이 왜 RLHF에서 중요한 관찰 포인트인지 말할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — RLHF 이전의 초기 policy가 어떻게 만들어지는지 다시 본다.
- [05_preference_optimization_dpo_orpo_kto](../05_preference_optimization_dpo_orpo_kto/README.md) — offline preference objective와 RLHF의 차이를 연결한다.
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — training instability와 regression check 감각을 복습한다.
- [05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto](../05_preference_optimization_dpo_orpo_kto/README.md) — rollout/update 규모가 커질 때 시스템 관점을 미리 떠올린다.
