# 05 Preference Optimization: DPO, ORPO, KTO 선행 개념

## 꼭 알고 오면 좋은 것
- SFT가 instruction-response example의 assistant 답변을 모방하도록 next-token loss를 주는 과정이라는 기본 감각
- token log-probability, negative log likelihood, log-prob margin을 읽을 수 있는 정도의 수학 감각
- train / validation / held-out eval을 분리해야 하며, offline win rate가 실제 사용자 만족을 보장하지 않는다는 이해
- chosen/rejected pair와 desirable/undesirable label이 서로 다른 supervision 구조라는 점
- reference policy 또는 anchor term이 policy drift를 줄이지만 behavior shift를 보수적으로 만들 수 있다는 직관
- alignment eval에서 helpfulness, factuality, safety/refusal, verbosity/style을 분리해야 한다는 관점

## 빠른 자기 점검
- chosen 응답이 절대 정답이 아니라 rejected보다 선호된 응답이라는 점을 설명할 수 있는가?
- `log p(chosen) - log p(rejected)`가 왜 preference objective의 최소 관찰값이 되는지 말할 수 있는가?
- DPO가 reference-relative margin을 보는 이유를 policy drift 관점으로 설명할 수 있는가?
- ORPO가 chosen likelihood anchor와 preference odds-ratio를 함께 본다는 설명을 받아들일 수 있는가?
- KTO가 strict pair 없이 desirable/undesirable label을 쓸 수 있지만 label noise에 민감하다는 점을 설명할 수 있는가?
- full RL loop 없이도 policy update를 할 수 있지만 alignment/eval tradeoff는 계속 남는다는 점을 이해하는가?

## 먼저 다시 보면 좋은 단위
- [04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — SFT policy가 preference optimization의 출발점이라는 점을 복습한다.
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — log-probability와 objective framing을 다시 확인한다.
- [03_domain_adaptive_pretraining](../03_domain_adaptive_pretraining/README.md) — policy drift와 guardrail을 distribution shift 관점으로 다시 본다.
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — loss 안정화와 eval regression 체크 감각을 복습한다.
