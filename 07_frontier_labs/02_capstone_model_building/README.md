# 02 Capstone Model Building

> Status: runnable
>
> 이 단위는 실제 외부 데이터셋, 네트워크, GPU 학습 없이 실행되는 **CPU-safe deterministic** capstone 설계 실습이다. 목표는 모델을 크게 돌리는 것이 아니라, 문제 정의·non-goal·dataset/model/eval contract·milestone·acceptance gate·risk register·failure-analysis/report outline을 실행 가능한 artifact로 고정하는 것이다.

## 왜 이 단위를 배우는가

앞선 트랙에서는 모델 구조, 학습 루프, 평가, 시스템 운영, 논문 재현 감각을 각각 익혔다. 하지만 실제 frontier 프로젝트에서는 그 조각들을 안다고 해서 곧바로 좋은 capstone이 생기지 않는다. 더 어려운 일은 **무엇을 만들지보다 무엇을 만들지 않을지, 어떤 성공 기준으로 끝낼지, 실패했을 때 무엇을 관찰할지를 먼저 계약으로 고정하는 것**이다.

이 단위는 막연한 "좋아 보이는 모델 아이디어"를 바로 구현으로 밀어붙이지 않는다. 대신 한국어 상품 검색 toy capstone을 예로 삼아 다음 질문을 artifact로 만든다.

- problem statement는 입력, 출력, baseline, 개선 기준을 한 문장으로 말하는가?
- non-goal은 serving, 신규 데이터 수집, personalization 같은 scope creep를 막는가?
- dataset contract는 split, schema, leakage control, label quality를 고정하는가?
- model contract는 baseline과 candidate를 같은 protocol에서 비교하게 만드는가?
- eval contract는 Recall@10, slice review, qualitative bucket을 함께 묶는가?
- milestone은 코드 TODO가 아니라 acceptance gate와 required artifact를 닫는가?
- risk register와 failure-analysis/report outline은 실패를 다음 실험으로 이어 주는가?

## 이번 단위에서 남길 것

- `scratch_lab.py` — capstone problem statement, non-goals, dataset/model/eval contract, milestone acceptance gates, risk register, failure-analysis outline을 `artifacts/scratch-manual/capstone_contract.json`으로 쓴다.
- `artifacts/scratch-manual/milestone_gates.svg` — M0~M3 gate 흐름을 보여 주는 작은 deterministic SVG다.
- `framework_lab.py` — framework-style project board, dataset-model-eval matrix, gate verdict, report outline, agentic loop handoff를 `artifacts/framework-manual/project_board.json`으로 쓴다.
- `analysis.py` — 두 artifact를 읽어 `artifacts/analysis-manual/latest_report.md`를 만든다. artifact가 없으면 먼저 실행할 명령을 알려 주며 실패한다.
- `analysis.md` — 실행마다 바뀌지 않는 stable interpretation 문서다.
- `reflection.md` — 실행 전 예측과 실행 후 scope/eval/failure 분석 질문이다.
- `lesson.yaml` — runnable 상태와 CPU-safe deterministic 실행 계약을 고정한다.

## 실행 방법

아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python3 07_frontier_labs/02_capstone_model_building/scratch_lab.py
python3 07_frontier_labs/02_capstone_model_building/framework_lab.py
python3 07_frontier_labs/02_capstone_model_building/analysis.py
```

생성되는 산출물은 다음 위치에 고정된다.

```text
07_frontier_labs/02_capstone_model_building/artifacts/
├── scratch-manual/
│   ├── capstone_contract.json
│   └── milestone_gates.svg
├── framework-manual/
│   └── project_board.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시

```text
$ python3 07_frontier_labs/02_capstone_model_building/scratch_lab.py
{
  "status": "runnable",
  "contract_type": "capstone_model_building_contract",
  "project_id": "korean_catalog_retrieval_capstone",
  "problem_statement": "한국어 상품 검색에서 text query와 image_caption을 입력으로 받아 lexical baseline 대비 Recall@10을 5pt 이상 개선한다.",
  "dataset_contract": {
    "source": "synthetic_korean_catalog_seed_v1",
    "split": {"train": 1200, "valid": 200, "test": 200}
  },
  "eval_contract": {
    "primary_metric": "Recall@10",
    "baseline_score": 0.42,
    "target_score": 0.49,
    "minimum_delta": 0.05
  }
}
```

```text
$ python3 07_frontier_labs/02_capstone_model_building/framework_lab.py
{
  "status": "runnable",
  "framework": "cpu_capstone_project_board_sim",
  "dataset_model_eval_matrix": {
    "dataset": "fixed synthetic split",
    "model_comparison": "lexical baseline vs tiny dual encoder",
    "eval_protocol": "Recall@10 + slice review"
  },
  "acceptance_gate_verdicts": [
    {"gate": "problem_scope_frozen", "verdict": "pass"},
    {"gate": "report_ready_with_failure_table", "verdict": "blocked_until_artifacts_complete"}
  ]
}
```

`analysis.py`는 artifact가 없으면 다음처럼 바로 고칠 수 있는 실패를 낸다.

```text
Missing required capstone artifact: ... Run scratch_lab.py and framework_lab.py first.
```

artifact가 있으면 `latest_report.md`에 problem statement, non-goals, dataset/model/eval contract, acceptance gates, risk register, failure-analysis outline, report outline을 요약한다.

## 실습 흐름

1. `scratch_lab.py`의 problem statement를 읽고 입력, 출력, baseline, metric이 모두 들어 있는지 확인한다.
2. non-goals를 읽고 이번 capstone이 일부러 하지 않는 일을 확인한다. 특히 real-time serving, external dataset collection, personalization을 제외해 scope boundary를 지킨다.
3. dataset contract에서 `synthetic_korean_catalog_seed_v1`, 고정 split, schema, leakage controls, label quality checks가 함께 있는지 본다.
4. model contract에서 `lexical_title_baseline`과 `tiny_dual_encoder` 후보가 같은 split/evaluator/query/failure bucket으로 비교되는지 확인한다.
5. eval contract에서 Recall@10 target delta와 secondary metric, qualitative bucket이 숫자와 해석을 같이 닫는지 본다.
6. milestone M0~M3를 읽고 각 단계가 acceptance gate와 required artifact를 갖는지 확인한다.
7. `framework_lab.py`의 project board를 읽고 gate verdict가 어디까지 pass/ready/not_started/blocked인지 구분한다.
8. `analysis.py` 보고서를 통해 이 contract가 다음 `07_frontier_labs/03_agentic_training_and_eval_loops`의 planner/verifier/critic handoff로 넘어갈 수 있는지 점검한다.

## 이 단위에서 특히 볼 질문

- problem statement는 실제로 끝낼 수 있는 한 프로젝트인가, 아니면 여러 프로젝트가 섞여 있는가?
- non-goal은 단순 겸손한 문장이 아니라 scope creep를 막는 운영 장치로 쓰이고 있는가?
- dataset / model / eval contract 셋 중 하나가 비어 있을 때 어떤 해석 문제가 생기는가?
- baseline은 가장 화려한 모델이 아니라 개선을 해석할 수 있는 가장 정직한 비교선인가?
- milestone은 코드 작업 목록인가, 아니면 의사결정과 artifact를 닫는 acceptance gate인가?
- risk register는 실패를 숨기는 문서인가, 아니면 다음 행동을 미리 정하는 triage 도구인가?
- failure-analysis/report outline을 먼저 쓰면 실험 중 어떤 로그를 반드시 남겨야 하는지가 더 선명해지는가?

## CPU/GPU 안전성

canonical path는 CPU-safe deterministic simulation이다. 실제 외부 dataset 다운로드, 네트워크 호출, GPU 학습, serving benchmark는 수행하지 않는다. 사용 가능한 GPU가 있어도 이 unit의 검증 기준은 동일한 JSON/SVG/Markdown artifact를 CPU에서 생성하는 것이다. GPU 실험은 이 contract를 바탕으로 별도 capstone run에서 선택적으로 붙일 수 있다.

## 다음 단위와의 연결

다음 단위 `07_frontier_labs/03_agentic_training_and_eval_loops`에서는 여기서 만든 capstone 계약을 planner / executor / verifier / critic loop로 넘긴다. 즉 agent가 실험을 도와주더라도, 먼저 **scope boundary, acceptance gate, failure slice, 보고서 구조**가 정리되어 있어야 자동화가 의미를 가진다. 이 단위에서 프로젝트 계약을 튼튼하게 잡아 두면, 다음 단위에서는 agentic loop를 단순 자동 실행기가 아니라 **검증 가능한 실험 운영자**로 붙일 수 있다.
