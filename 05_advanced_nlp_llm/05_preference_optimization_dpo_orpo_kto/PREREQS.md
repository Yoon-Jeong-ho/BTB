# 05 Preference Optimization: DPO, ORPO, KTO 선행 개념

## 꼭 알고 오면 좋은 것
- supervised fine-tuning(SFT)이 prompt-response likelihood를 최대화하는 과정이라는 기본 감각
- token log-probability, cross entropy, likelihood ratio를 아주 거칠게라도 읽을 수 있는 정도의 수학 감각
- train / validation / held-out evaluation 분리와 data leakage의 위험
- pairwise comparison과 binary label이 서로 다른 supervision 신호라는 점
- reference model / KL regularization이 왜 policy drift를 막는 장치로 자주 등장하는지에 대한 직관
- alignment 평가에서 win rate, factuality, safety, verbosity가 서로 다른 축이라는 이해

## 빠른 자기 점검
- chosen / rejected pair는 "정답 / 오답"과 같지 않다는 점을 설명할 수 있는가?
- SFT objective와 preference objective가 각각 무엇을 직접 밀어 올리는지 구분할 수 있는가?
- reference policy를 두는 이유를 "너무 멀리 가지 않게 하는 기준점" 정도로 설명할 수 있는가?
- pairwise preference 데이터가 없을 때 desirable / undesirable label 기반 학습이 왜 대안이 될 수 있는지 말할 수 있는가?
- offline judge win rate가 length bias, style bias, safety regression을 가릴 수 있다는 점을 받아들일 수 있는가?

## 먼저 다시 보면 좋은 단위
- [04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — SFT policy를 preference optimization의 출발점으로 다시 본다.
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — log-prob와 objective framing 감각을 다시 연결한다.
- [01_ml/03_model_selection_and_interpretation](../../01_ml/03_model_selection_and_interpretation/README.md) — 지표 해석과 validation 분리 감각을 복습한다.
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — loss 안정화와 evaluation regression 체크 감각을 복습한다.
