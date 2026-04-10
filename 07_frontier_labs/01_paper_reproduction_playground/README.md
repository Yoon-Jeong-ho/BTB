# 01 Paper Reproduction Playground

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
논문을 읽을 때 가장 흔한 착각은 "저자들이 한 일을 전부 다시 해야 재현"이라고 생각하는 것이다. 실제 연구 운영에서는 그보다 먼저 **무슨 claim을 어디까지 믿고 싶고, 그 claim을 어떤 evidence로 다시 확인할 것인가**를 좁혀야 한다. 이 단위는 paper reproduction을 거대한 복제 프로젝트가 아니라, **claim → baseline → metric → artifact**로 이어지는 작은 실험 계약으로 바꾸는 출발점이다.

또한 이 감각이 있어야 다음 frontier 단위들에서 capstone을 설계하거나 agentic 실험 루프를 만들 때도 "논문 느낌"이 아니라 **재현 가능한 비교 기준과 관찰 로그**를 남길 수 있다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- reproduction scope control, claim/evidence mindset, baseline 비교 원칙을 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - paper claim → experiment mapping 표
  - baseline / reported / reproduced metric 비교표
  - seed / split / hardware / runtime 로그 요약
  - mismatch hypothesis와 failure observation 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable/applied 승격 때 구현할 실습 순서다.
1. 먼저 논문 한 편을 통째로 들고 오기보다, **재현해 보고 싶은 핵심 claim 1~2개**를 고른다. 예를 들어 성능 향상, ablation 결론, scaling trend, failure mode 설명 중 무엇이 핵심인지 먼저 자른다.
2. 그다음 최소 재현 범위를 정한다. dataset 전체를 다 쓸지, 작은 subset으로 trend만 볼지, full training 대신 frozen baseline + 짧은 fine-tuning으로 갈지처럼 compute/time budget 안에서 scope를 잠근다.
3. baseline을 다시 세운다. 논문이 보고한 숫자만 보는 것이 아니라, **같은 metric 정의 / 같은 split / 같은 preprocessing 계약**으로 비교 가능한 baseline을 무엇으로 둘지 명시한다.
4. 실험 카드(experiment card)를 만든다. claim, expected evidence, required artifact, 실패 시 관찰 포인트를 미리 적어 두고, run이 끝난 뒤 무엇을 "성공" 또는 "불일치"로 볼지 기준을 정한다.
5. 실행 단계에서는 reported result와 reproduced result를 숫자 하나로만 비교하지 않고, variance, seed sensitivity, runtime 제약, preprocessing 차이, hidden trick 가능성을 함께 기록한다.
6. 마지막에는 reproduction log를 정리해 다음 단위 `07_frontier_labs/02_capstone_model_building`으로 넘긴다. 즉 논문을 다시 따라 한 기록에서 끝내지 않고, **무엇이 내 프로젝트 설계 원칙으로 남는가**를 문장으로 남긴다.

## 이 단위에서 특히 볼 질문
- "논문을 재현한다"는 말은 full paper 복제인가, 아니면 특정 claim을 검증 가능한 실험으로 다시 세우는 일인가?
- compute, 시간, 데이터 접근 제약이 있을 때 reproduction scope는 어디까지 줄여도 정직하다고 말할 수 있는가?
- baseline은 논문 표에 적힌 숫자를 가져오는 것으로 충분한가, 아니면 같은 protocol 아래에서 다시 돌린 비교선이 필요한가?
- reported result와 reproduced result가 다를 때, 먼저 의심해야 할 것은 구현 실수인가, preprocessing 차이인가, variance인가, 숨은 실험 조건인가?
- claim/evidence 관점에서 보면 "숫자가 비슷했다"보다 더 중요한 관찰은 무엇인가?
- reproduction artifact를 남길 때 어떤 로그와 메타데이터가 있어야 다음 단위의 capstone 실험으로 자연스럽게 이어질 수 있는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py
{
  "status": "sample",
  "paper": "example-paper-2024",
  "claim_id": "C1",
  "claim": "Adapter tuning reduces compute while preserving 95%+ of full fine-tune accuracy.",
  "scope": {
    "task": "classification",
    "dataset_slice": "train[:20%] / val[:20%]",
    "budget_note": "1 GPU day 이하 playground 재현"
  },
  "baseline": {
    "name": "full_finetune_small",
    "metric": {"accuracy": 0.842}
  },
  "reported": {
    "metric": {"accuracy": 0.851},
    "notes": ["paper table 2", "3 seeds averaged"]
  },
  "reproduced": {
    "metric": {"accuracy": 0.833},
    "delta_vs_reported": -0.018,
    "delta_vs_baseline": -0.009
  },
  "artifact_check": {
    "seed_logged": true,
    "split_logged": true,
    "hardware_logged": true,
    "missing": ["paper preprocessing detail"]
  }
}

$ python 07_frontier_labs/01_paper_reproduction_playground/analysis.py
{
  "status": "sample",
  "claim_evidence_matrix_shape": [3, 5],
  "observations": [
    "baseline 재실행 결과가 paper appendix 숫자보다 낮음",
    "tokenization/preprocessing 차이가 주요 mismatch 후보",
    "seed variance가 reported delta와 같은 크기"
  ],
  "next_actions": [
    "preprocessing alignment 확인",
    "same-eval protocol 재검증",
    "capstone용 reusable experiment template 추출"
  ]
}
```

핵심은 논문 표의 숫자를 외우는 것이 아니라, **어떤 claim을 어떤 evidence로 다시 세웠는지**, **baseline과 reproduced run을 같은 조건에서 비교했는지**, **불일치가 났을 때 무엇을 관찰 로그로 남겼는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/02_capstone_model_building`에서는 이제 남의 논문 claim을 따라가는 데서 한 걸음 더 나아가, **내가 만들 모델/실험의 성공 기준과 비교 기준을 직접 설계**해야 한다. 그래서 이 단위에서 reproduction scope control, baseline hygiene, claim/evidence logging 습관을 먼저 잡아 두면, capstone 단계에서도 막연한 구현 대신 **검증 가능한 프로젝트 계약**으로 바로 이어질 수 있다.
