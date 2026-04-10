# 03 Agentic Training and Eval Loops 선행 개념

## 꼭 알고 오면 좋은 것
- capstone scope, acceptance gate, non-goal을 먼저 고정해야 자동화가 의미 있다는 이해
- baseline / metric / split / failure slice를 같은 조건에서 비교해야 한다는 기본 실험 운영 태도
- train/eval run에서 seed, config, hardware, runtime log가 빠지면 결과 해석이 흔들린다는 감각
- retry policy와 stop condition이 없으면 compute만 늘고 정보량은 줄 수 있다는 문제의식
- 평가 숫자 하나보다 artifact completeness와 protocol match가 더 먼저 확인돼야 한다는 이해
- 사람 검토가 필요한 escalation 지점을 일부러 남겨 두는 것이 agentic workflow 품질에 중요하다는 점

## 먼저 다시 보면 좋은 단위
- [07_frontier_labs/02_capstone_model_building](../02_capstone_model_building/README.md) — experiment contract, milestone, acceptance gate를 agent loop 입력으로 넘기는 감각을 복습한다.
- [07_frontier_labs/01_paper_reproduction_playground](../01_paper_reproduction_playground/README.md) — claim/evidence와 baseline comparability를 어떻게 기록해야 하는지 다시 확인한다.
- [05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval](../../05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/README.md) — evaluation protocol과 grounded evidence를 loop gate 관점에서 다시 본다.
- [05_advanced_nlp_llm/06_rlhf_and_reasoning_rl](../../05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/README.md) — verifier / critic / reward-like feedback가 어떻게 학습 루프에 들어오는지 연결해서 본다.
- [06_training_systems/09_profiling_monitoring_and_failure_recovery](../../06_training_systems/09_profiling_monitoring_and_failure_recovery/README.md) — runtime 관측, failure recovery, artifact logging을 실제 운영 루프 관점에서 복습한다.

## 빠른 자기 점검
- capstone 실험 계약에서 무엇을 planner가 바꿔도 되고 무엇은 고정해야 하는지 말할 수 있는가?
- planner / executor / verifier / critic 역할을 왜 분리해야 하는지, 각 역할이 남겨야 할 artifact를 예로 들 수 있는가?
- metric 개선이 있어도 verifier가 통과시키면 안 되는 상황을 두세 가지 떠올릴 수 있는가?
- retry budget과 stop condition이 없을 때 어떤 종류의 loop failure가 생기는지 설명할 수 있는가?
- critic recommendation이 추측인지 evidence 기반 triage인지 구분하는 기준을 말할 수 있는가?
- benchmark drift나 contamination 의심이 생겼을 때, loop를 더 돌리는 대신 사람이 개입해야 하는 이유를 설명할 수 있는가?
