# Prerequisites

이 단위에 들어가기 전에는 아래를 먼저 확인한다.

1. `08_multimodal_bridge/01_contrastive_alignment` — 이미지와 텍스트 표현을 같은 공간에 놓는 감각
2. `09_multimodal/03_visual_question_answering` — 시각 정보와 질문을 함께 읽는 감각
3. `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl` — reward, policy, rollout, KL 같은 RLHF 용어
4. `02_deep_learning/04_attention_and_transformers` — 여러 token/feature를 섞어 decision head로 보내는 감각
5. [RL to VLA bridge](../../docs/08_rl_to_vla_bridge.md) — MDP, trajectory, behavior cloning, offline RL, action space design을 RLHF 용어와 구분하는 감각

## 체크 질문

- VQA answer와 action token의 차이를 설명할 수 있는가?
- action accuracy와 safety violation이 서로 충돌할 수 있는 예를 만들 수 있는가?
- 실제 로봇/시뮬레이터 실험에서 trajectory log가 왜 필요한지 말할 수 있는가?
- behavior cloning과 offline RL이 같은 demonstration 데이터를 다르게 쓰는 이유를 말할 수 있는가?
