# 02 Capstone Model Building 선행 개념

## 꼭 알고 오면 좋은 것
- 막연한 아이디어를 baseline / metric / artifact가 있는 작은 실험 계약으로 자르는 감각
- dataset split, leakage, label noise, slice evaluation이 결과 해석에 미치는 영향 이해
- baseline과 candidate model을 같은 protocol에서 비교해야 한다는 기본 태도
- runtime budget, hardware 제약, logging, failure recovery가 프로젝트 범위를 바꿀 수 있다는 점
- 결과 숫자 하나보다 report structure와 failure analysis가 더 오래 남는 산출물이라는 문제의식
- success criterion과 non-goal을 함께 적어야 scope creep를 막을 수 있다는 이해

## 먼저 다시 보면 좋은 단위
- [07_frontier_labs/01_paper_reproduction_playground](../01_paper_reproduction_playground/README.md) — claim/evidence reproduction 로그를 capstone 프로젝트 계약으로 넘기는 감각을 복습한다.
- [05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval](../../05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/README.md) — evaluation protocol, grounded evidence, failure slice 설계를 다시 묶는다.
- [05_advanced_nlp_llm/08_alignment_safety_and_model_behavior](../../05_advanced_nlp_llm/08_alignment_safety_and_model_behavior/README.md) — behavior boundary와 safety-oriented qualitative review 감각을 capstone scope에 반영한다.
- [06_training_systems/09_profiling_monitoring_and_failure_recovery](../../06_training_systems/09_profiling_monitoring_and_failure_recovery/README.md) — runtime budget, observability, 실패 복구를 프로젝트 milestone에 넣는 법을 복습한다.
- [07_frontier_labs/03_agentic_training_and_eval_loops](../03_agentic_training_and_eval_loops/README.md) — milestone과 failure slice를 agentic 실행 계약으로 넘기는 감각을 미리 염두에 둔다.

## 빠른 자기 점검
- 내가 하고 싶은 capstone 아이디어를 한 문장의 problem statement와 한 문장의 non-goal로 동시에 적을 수 있는가?
- baseline을 왜 가장 강한 모델이 아니라 가장 해석 가능한 비교선에서 시작해야 하는지 설명할 수 있는가?
- dataset / model / eval 계약 중 하나가 비어 있을 때 어떤 해석 문제가 생기는지 예를 들 수 있는가?
- milestone을 코드 작업 목록이 아니라 의사결정 종료 기준과 artifact 계약으로 쓸 수 있는가?
- failure analysis를 마지막 부록이 아니라 처음부터 설계해야 하는 이유를 말할 수 있는가?
- 다음 사람이 내 프로젝트를 이어받았을 때 바로 이해할 수 있도록 report outline과 risk log에 무엇이 들어가야 하는지 떠올릴 수 있는가?
