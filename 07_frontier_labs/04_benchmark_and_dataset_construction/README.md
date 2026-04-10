# 04 Benchmark and Dataset Construction

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
앞 단위에서 agentic loop와 verifier gate를 세웠더라도, 그 loop가 최적화하는 benchmark와 dataset contract가 약하면 결국 자동화된 속도로 잘못된 신호를 키우게 된다. 그래서 frontier 실험에서 benchmark construction은 뒤늦게 붙이는 리더보드 장식이 아니라, **무엇을 성공으로 읽을지와 무엇을 아직 주장할 수 없는지를 함께 고정하는 측정 계약**에 가깝다.

또한 dataset construction은 단순 수집 작업이 아니다. 어떤 샘플을 한 개 사례로 볼지, split을 어디서 끊을지, annotation disagreement를 어떻게 처리할지, contamination과 drift를 어떻게 감시할지까지 정해야 비로소 benchmark가 실험 운영의 기준점이 된다. 이 단위는 benchmark/dataset을 모델 바깥의 부속물이 아니라 **연구 운영 전체를 정직하게 만드는 인터페이스**로 읽게 만드는 것이 목적이다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- benchmark mindset, dataset contract, annotation/QC, leakage/drift 위험을 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - benchmark card와 task contract 요약
  - dataset schema / source / split manifest
  - annotation rubric, adjudication rule, quality-control report
  - contamination / leakage / drift watchlist
  - final reporting template와 benchmark version note

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable/applied 승격 때 구현할 실습 순서다.
1. 먼저 이전 capstone과 agentic loop에서 어떤 metric과 failure slice를 쓰고 있었는지 다시 읽는다. benchmark를 새로 만든다는 말은 점수판을 하나 더 만드는 것이 아니라, **무슨 신호를 믿을지 다시 계약하는 일**이기 때문이다.
2. task contract를 적는다. 입력은 무엇인지, 출력은 어떤 형식인지, 평가 대상 단위(unit of record)는 문서인지 대화 turn인지 trajectory인지, 정답/선호/행동 평가 중 무엇을 보려는지 먼저 못 박는다.
3. dataset contract를 만든다. source, license, collection boundary, dedup policy, schema, missing value 처리, metadata 필드를 정리하고, 어떤 샘플이 benchmark에 들어올 자격이 있는지와 빠져야 하는지를 적는다.
4. split hygiene를 설계한다. random split으로 끝내지 않고 user/source/time/topic/template 단위 leakage를 막기 위해 어떤 축에서 분리할지 정하고, public/dev/test를 언제 freeze할지 결정한다.
5. annotation plan을 만든다. rubric, label taxonomy, multi-annotator overlap, adjudication rule, abstain 정책, ambiguity logging 방식을 정해 label quality를 숫자와 메모 둘 다로 읽을 수 있게 한다.
6. eval protocol을 고정한다. primary metric, slice metric, human spot-check bucket, judge/evaluator 사용 여부, contamination audit, drift monitoring 기준을 묶어 **benchmark를 운영하는 절차**를 만든다.
7. 마지막에는 benchmark report template를 먼저 적고 다음 단위 `07_frontier_labs/05_open_ended_research_tracks`로 넘긴다. 그래야 이후 열린 연구 질문도 benchmark를 흔드는 방향인지, benchmark 위에서 정직하게 비교하는 방향인지 구분할 수 있다.

## 이 단위에서 특히 볼 질문
- benchmark는 단순 leaderboard인가, 아니면 연구팀이 무엇을 성공으로 주장할 수 있는지 정하는 계약인가?
- dataset contract에서 정말 고정해야 하는 것은 schema뿐인가, 아니면 unit of record, source boundary, licensing, split freeze 시점까지 포함하는가?
- split hygiene는 random split만 잘하면 충분한가, 아니면 user/source/time/template 단위 near-duplicate와 contamination까지 봐야 하는가?
- annotation disagreement는 noise인가, 아니면 task definition이 약하다는 신호인가?
- benchmark score가 올랐을 때 그것이 진짜 capability improvement인지, evaluator overfitting / contamination / drift인지 어떻게 구분할 수 있는가?
- benchmark를 자주 업데이트하면 현실 적합성이 좋아질 수 있지만, 동시에 비교 가능성은 어떻게 흔들리는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 07_frontier_labs/04_benchmark_and_dataset_construction/scratch_lab.py
{
  "status": "sample",
  "benchmark_id": "btb-agent-eval-v1",
  "task_contract": {
    "input_unit": "trajectory",
    "target": "tool-using assistant response",
    "primary_claim": "benchmark가 planning, tool grounding, safety refusal를 함께 구분해 측정한다"
  },
  "dataset_contract": {
    "sources": ["internal_eval_logs_v2", "synthetic_scenarios_v1", "manual_redteam_v1"],
    "schema_fields": ["prompt", "context", "reference", "slice_tags", "source_id"],
    "license_notes": ["internal only", "manual release review required"]
  },
  "splits": {
    "dev": 1200,
    "test_public": 600,
    "test_private": 600,
    "split_rule": ["source_id disjoint", "template family disjoint", "recent data held out by time"]
  },
  "annotation": {
    "rubric_dimensions": ["task_success", "groundedness", "policy_compliance"],
    "double_label_rate": 0.25,
    "adjudication": "expert_review_if_major_disagreement"
  },
  "quality_gates": {
    "invalid_schema_rate_max": 0.01,
    "annotator_agreement_floor": 0.72,
    "near_duplicate_alert_threshold": 0.03
  }
}

$ python 07_frontier_labs/04_benchmark_and_dataset_construction/analysis.py
{
  "status": "sample",
  "contamination_audit": {
    "exact_overlap_hits": 0,
    "near_duplicate_hits": 7,
    "judge_prompt_leakage_flags": 2
  },
  "drift_watchlist": [
    "최근 2주 데이터에서 agent tool schema 변경 발생",
    "safety refusal label 기준이 annotator cohort마다 다르게 적용됨"
  ],
  "report_sections": [
    "task_contract",
    "dataset_contract",
    "split_hygiene",
    "annotation_qc",
    "eval_protocol",
    "known_limits"
  ],
  "next_actions": [
    "template-family split 재검토",
    "ambiguous labels adjudication 사례집 추가",
    "private holdout contamination audit 자동화"
  ]
}
```

중요한 것은 benchmark 이름을 붙이는 일이 아니라, **무엇을 측정하고 무엇은 아직 못 측정하는지**, **dataset contract와 split이 정말 비교 가능한지를 보장하는지**, **annotation과 contamination audit가 score 해석을 뒷받침하는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/05_open_ended_research_tracks`에서는 이제 benchmark가 고정된 뒤, 정답이 없는 연구 질문을 어떤 stopping rule과 evidence 기준으로 쪼갤지 다룬다. 이 단위에서 benchmark contract, split hygiene, QC gate를 먼저 세워 두면 다음 연구 트랙에서는 막연히 새 아이디어를 많이 시도하는 대신, **무엇이 실제 개선이고 무엇이 benchmark 착시인지**를 더 빠르게 구분할 수 있다.
