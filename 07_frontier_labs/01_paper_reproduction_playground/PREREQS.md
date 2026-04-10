# 01 Paper Reproduction Playground 선행 개념

## 꼭 알고 오면 좋은 것
- baseline, metric, validation split을 같은 조건에서 비교해야 한다는 기본 모델 비교 감각
- learning curve, seed, batch, runtime budget이 결과 해석에 영향을 준다는 실험 운영 기본기
- 논문 표의 숫자를 읽을 때 absolute result와 relative trend를 구분하는 습관
- preprocessing / evaluator / hardware 차이가 재현성에 영향을 줄 수 있다는 이해
- 실험 로그를 남기지 않으면 mismatch 원인을 설명하기 어렵다는 점
- "구현 성공"과 "claim 검증 성공"이 같은 말은 아니라는 문제의식

## 먼저 다시 보면 좋은 단위
- [01_ml/03_model_selection_and_interpretation](../../01_ml/03_model_selection_and_interpretation/README.md) — baseline 비교와 metric 해석 습관 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — learning curve, seed, debugging log를 읽는 기본기 복습
- [05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval](../../05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/README.md) — claim/evidence와 grounded evaluation을 구분하는 태도 연결
- [06_training_systems/09_profiling_monitoring_and_failure_recovery](../../06_training_systems/09_profiling_monitoring_and_failure_recovery/README.md) — runtime / hardware / failure observation을 artifact로 남기는 습관 복습

## 빠른 자기 점검
- 논문 전체를 복제하는 것과 핵심 claim 하나를 정직하게 재현하는 것의 차이를 설명할 수 있는가?
- baseline 숫자를 paper 표에서 가져오는 것만으로는 왜 충분하지 않은지 말할 수 있는가?
- reproduced result가 reported result보다 낮게 나왔을 때, 구현 버그 외에 어떤 mismatch 후보를 먼저 적을지 떠올릴 수 있는가?
- seed variance와 evaluation protocol 차이가 작은 성능 차이를 뒤집을 수 있다는 점을 받아들일 수 있는가?
- 다음 사람이 내 실험을 이어서 볼 수 있도록 어떤 metadata를 남겨야 하는지 최소한의 목록을 말할 수 있는가?
