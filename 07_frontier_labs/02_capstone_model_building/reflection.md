# 02 Capstone Model Building 회고

## 실행 전 예측

- problem statement 안에 입력, 출력, baseline, metric, target delta가 모두 들어 있는지 먼저 확인한다.
- non-goal이 최소 세 가지 이상인지 확인한다. 특히 real-time serving, external dataset collection, personalization 같은 scope creep 후보가 빠져 있어야 한다.
- dataset / model / eval contract 중 하나라도 비면 capstone 결과를 해석하기 어렵다고 예측한다.
- milestone M0~M3는 코드 TODO가 아니라 acceptance gate와 required artifact를 닫는 흐름이어야 한다.

## 실행 후 관찰

- `scratch_lab.py`가 생성한 contract에서 dataset split과 leakage controls가 baseline/model/eval 비교의 전제인지 확인한다.
- `framework_lab.py`의 project board에서 어떤 gate가 pass, ready, not_started, blocked인지 구분한다.
- final gate가 `blocked_until_artifacts_complete`로 남는 이유를 설명한다. 이 unit은 실제 모델 run을 꾸며 내지 않고, artifact가 있어야만 최종 보고서가 닫힌다는 사실을 드러낸다.
- `analysis.py` 보고서가 risk register와 failure-analysis outline을 final report 구조로 연결하는지 확인한다.

## 아직 애매하면 다시 볼 질문

- 내 capstone idea가 problem statement 한 문장으로 줄어들지 않는다면, 어떤 sub-project를 분리해야 하는가?
- baseline이 약해서 improvement가 과장되는 상황과, baseline이 너무 강해 toy unit에서 학습 신호가 사라지는 상황을 어떻게 구분할 수 있는가?
- Recall@10이 올라갔지만 brand_mismatch slice가 나빠졌다면 M2 gate를 통과시켜야 하는가?
- risk register의 mitigation은 실제 다음 행동인가, 아니면 단순한 걱정 목록인가?
- failure analysis는 결과가 나쁠 때 쓰는 부록인가, 아니면 실험 전부터 table shape를 정해야 하는 core deliverable인가?

## 다음에 다시 볼 것

- `07_frontier_labs/03_agentic_training_and_eval_loops`에서 planner / executor / verifier / critic 역할이 이 capstone contract를 어떻게 읽는지 확인한다.
- `07_frontier_labs/04_benchmark_and_dataset_construction`에서 dataset contract와 eval contract가 더 엄격한 benchmark design으로 어떻게 확장되는지 확인한다.
