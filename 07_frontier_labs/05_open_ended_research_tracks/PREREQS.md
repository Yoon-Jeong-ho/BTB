# 05 Open-Ended Research Tracks 선행 개념

## 꼭 알고 오면 좋은 것
- capstone scope, benchmark contract, evaluation protocol을 문장으로 먼저 고정해야 탐색이 정직해진다는 감각
- agentic loop에서 planner / executor / verifier / critic 역할을 나눠야 self-approval를 줄일 수 있다는 이해
- benchmark drift, contamination, protocol mismatch가 좋아 보이는 결과도 보류하게 만들 수 있다는 문제의식
- baseline-relative comparison, failure slice observation, qualitative memo를 함께 남겨야 exploratory run이 해석 가능해진다는 인식
- retry budget, stopping rule, archive note가 연구 생산성을 꺾는 장치가 아니라 wandering을 막는 장치라는 이해
- negative result와 inconclusive result를 구분해 기록해야 다음 행동이 달라진다는 기본 운영 감각

## 먼저 다시 보면 좋은 단위
- [07_frontier_labs/01_paper_reproduction_playground](../01_paper_reproduction_playground/README.md) — reproduction scope와 claim boundary를 어디까지 좁혀야 하는지 다시 읽는다.
- [07_frontier_labs/02_capstone_model_building](../02_capstone_model_building/README.md) — north-star goal과 this-project scope를 분리하는 감각을 복습한다.
- [07_frontier_labs/03_agentic_training_and_eval_loops](../03_agentic_training_and_eval_loops/README.md) — iteration contract, retry budget, verifier/critic gate를 exploratory track 운영에 다시 연결한다.
- [07_frontier_labs/04_benchmark_and_dataset_construction](../04_benchmark_and_dataset_construction/README.md) — benchmark contract와 evidence trust가 흔들릴 때 왜 research track도 함께 멈춰야 하는지 확인한다.
- [06_training_systems/09_profiling_monitoring_and_failure_recovery](../../06_training_systems/09_profiling_monitoring_and_failure_recovery/README.md) — 실험 실패를 운영 신호로 읽고 기록하는 runbook 감각을 복습한다.

## 빠른 자기 점검
- open-ended research와 scope-less wandering의 차이를 한두 문장으로 설명할 수 있는가?
- hypothesis를 적을 때 claim, boundary, evidence, kill criterion을 왜 같이 적어야 하는지 말할 수 있는가?
- exploratory phase에서도 baseline-relative signal과 negative result 기록이 왜 필요한지 설명할 수 있는가?
- no-signal과 inconclusive result를 구분하지 않으면 어떤 잘못된 후속 판단이 생기는지 예로 들 수 있는가?
- stopping rule이 없는 끈질긴 탐색과 잘 설계된 persistent research의 차이를 설명할 수 있는가?
- archive note가 성공 사례뿐 아니라 중단 이유와 재개 조건까지 남겨야 하는 이유를 말할 수 있는가?
