# 02 Capstone Model Building

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
앞선 트랙에서는 모델 구조, 학습 루프, 평가, 시스템 운영, 논문 재현 감각을 각각 익혔다. 하지만 실제 frontier 프로젝트에서는 그 조각들을 안다고 해서 곧바로 좋은 capstone이 생기지 않는다. 더 어려운 일은 **무엇을 만들지보다 무엇을 만들지 않을지, 어떤 성공 기준으로 끝낼지, 실패했을 때 무엇을 관찰할지를 먼저 계약으로 고정하는 것**이다.

이 단위는 막연한 "좋아 보이는 모델 아이디어"를 바로 구현으로 밀어붙이기보다, **문제 정의 → 데이터 계약 → 모델 비교선 → 평가 기준 → milestone → 보고서 구조**로 이어지는 end-to-end 프로젝트 뼈대를 설계하는 연습이다. 즉, 프로젝트를 시작하기 전에 이미 절반을 설명 가능한 상태로 만드는 문서화 습관을 잡는다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- capstone scoping, milestone decomposition, dataset/model/eval framing 원칙을 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - capstone problem statement와 non-goal 정의
  - milestone별 acceptance gate와 risk register
  - dataset / model / eval matrix와 baseline 비교표
  - report outline, failure analysis table, next-step memo

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable/applied 승격 때 구현할 실습 순서다.
1. 먼저 capstone 질문을 하나로 줄인다. "멀티모달 agent를 만든다"처럼 넓은 문장이 아니라, **누구의 어떤 입력을 받아 어떤 출력 품질을 개선할 것인가**를 한 문장으로 못 박는다.
2. 그다음 scope boundary를 적는다. 무엇을 이번 프로젝트의 성공으로 볼지와 함께, **이번 프로젝트에서 일부러 하지 않을 것**도 적는다. 그래야 milestone이 끝없이 늘어나지 않는다.
3. dataset contract를 만든다. 어떤 데이터 소스를 쓸지, split은 어떻게 고정할지, label 품질과 leakage 위험은 무엇인지, 추가 수집/정제가 필요한지 먼저 적는다.
4. model contract를 만든다. 가장 먼저 비교할 baseline이 무엇인지, candidate model family는 무엇인지, fine-tuning / frozen / retrieval / distillation 중 어떤 경로를 우선 볼지, compute budget은 어느 정도인지 정한다.
5. eval contract를 만든다. 주 metric, 보조 metric, slice evaluation, 사람이 직접 확인할 qualitative bucket, failure case 기록 방식을 정해 **좋은 결과와 나쁜 결과를 같은 언어로 읽을 수 있게** 만든다.
6. milestone을 분해한다. 예를 들어 M0는 data/baseline 정리, M1은 최소 파이프라인, M2는 개선 실험, M3는 failure analysis와 report 정리처럼 끊고, 각 단계의 종료 조건과 산출물을 명시한다.
7. 마지막에는 보고서 뼈대를 먼저 쓴다. 즉 실험이 끝난 뒤 무엇을 적을지가 아니라, **실험하면서 반드시 남겨야 할 표/로그/질문**을 미리 정한 뒤 다음 단위 `07_frontier_labs/03_agentic_training_and_eval_loops`로 넘긴다.

## 이 단위에서 특히 볼 질문
- 지금 생각한 capstone scope는 실제로 끝낼 수 있는 문제인가, 아니면 아직도 여러 프로젝트가 섞여 있는가?
- baseline은 가장 화려한 모델이 아니라, 이후 개선을 해석할 수 있는 **가장 정직한 비교선**으로 잡혀 있는가?
- dataset / model / eval 셋 중 하나가 비어 있을 때 어떤 종류의 혼란이 생기는가?
- milestone은 코드 작업 목록인가, 아니면 **의사결정과 artifact를 닫는 관문**인가?
- 실패 분석은 결과가 나쁠 때만 쓰는 부록인가, 아니면 처음부터 slice와 로그가 정의되어야 하는 core deliverable인가?
- 프로젝트 보고서를 나중에 꾸미는 문서가 아니라, 실험 중 관찰을 묶는 operating document로 볼 수 있는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 07_frontier_labs/02_capstone_model_building/scratch_lab.py
{
  "status": "sample",
  "project_id": "capstone-mm-retrieval-v1",
  "problem_statement": "한국어 상품 이미지-텍스트 검색 품질을 baseline 대비 개선한다.",
  "non_goals": [
    "실시간 serving 최적화는 이번 범위에서 제외",
    "새 데이터셋 수집은 최소 수작업 검증만 수행"
  ],
  "dataset_contract": {
    "source": ["internal_catalog_v2", "manual_eval_set_v1"],
    "split": {"train": 120000, "valid": 5000, "test": 5000},
    "risks": ["중복 상품 노출", "라벨 noise", "카테고리 불균형"]
  },
  "model_contract": {
    "baseline": "dual-encoder-small",
    "candidate": ["larger_dual_encoder", "reranker_head"],
    "budget_note": "2 x A100 1일 이내"
  },
  "eval_contract": {
    "primary_metric": "Recall@10",
    "secondary_metrics": ["MRR", "category_slice_recall"],
    "qualitative_buckets": ["fine-grained confusion", "brand mismatch", "OCR failure"]
  },
  "milestones": [
    {"id": "M0", "goal": "dataset/baseline contract freeze"},
    {"id": "M1", "goal": "minimal reproducible training/eval pipeline"},
    {"id": "M2", "goal": "improvement experiment + ablation"},
    {"id": "M3", "goal": "failure analysis + final report"}
  ]
}

$ python 07_frontier_labs/02_capstone_model_building/analysis.py
{
  "status": "sample",
  "report_sections": [
    "problem_and_scope",
    "dataset_contract",
    "baseline_and_model_choices",
    "eval_protocol",
    "results",
    "failure_analysis",
    "next_steps"
  ],
  "open_risks": [
    "validation split leakage 의심",
    "baseline 재현 수치 불안정",
    "qualitative failure tag taxonomy 미정"
  ],
  "next_actions": [
    "M0 acceptance gate 문장화",
    "failure slice 표준화",
    "agentic retry loop에 넣을 triage 규칙 정의"
  ]
}
```

중요한 것은 모델 이름을 멋지게 붙이는 일이 아니라, **문제·데이터·모델·평가·실패 분석이 한 문서 계약 안에서 서로 물려 있는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/03_agentic_training_and_eval_loops`에서는 여기서 만든 capstone 계약을 실제 실행·triage·재평가 루프로 넘긴다. 즉 agent가 실험을 도와주더라도, 먼저 **scope boundary, acceptance gate, failure slice, 보고서 구조**가 정리되어 있어야 자동화가 의미를 가진다. 이 단위에서 프로젝트 계약을 튼튼하게 잡아 두면, 다음 단위에서는 agentic loop를 단순 자동 실행기가 아니라 **검증 가능한 실험 운영자**로 붙일 수 있다.
