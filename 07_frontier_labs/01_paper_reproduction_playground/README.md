# 01 Paper Reproduction Playground

> Status: runnable
>
> 이 단위는 CPU에서 바로 실행되는 CPU-safe deterministic paper reproduction 실습이다. 실제 논문 다운로드, 네트워크 호출, GPU 학습 없이 claim/evidence matrix, baseline/reported/reproduced comparison, scope control, variance, mismatch hypothesis, artifact hygiene를 작은 toy reproduction 계약으로 관찰한다.

## 왜 이 단위를 배우는가
논문 재현을 처음 시작하면 “논문 전체를 다시 구현해야 한다”거나 “표의 숫자를 비슷하게 맞추면 끝”이라고 생각하기 쉽다. 하지만 실제 연구 운영에서 먼저 필요한 것은 **어떤 claim을 어느 범위까지 다시 확인할지**와 **그 claim을 어떤 evidence로 판단할지**를 좁히는 일이다.

이 runnable 단위는 paper reproduction을 거대한 복제 프로젝트가 아니라, `claim_id → evidence → comparison → mismatch hypothesis → artifact` 흐름으로 바꾸는 연습이다. baseline, reported, reproduced 숫자를 모두 보되 primary comparison은 같은 protocol에서 다시 돌린 reproduced baseline vs reproduced method로 제한한다. 또한 숫자가 어긋났을 때 variance, preprocessing alignment, evaluator mismatch, budget mismatch를 관찰 로그로 남긴다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: embedded toy paper card로 claim/evidence matrix, baseline/reported/reproduced comparison, variance summary, mismatch hypothesis, `paper_reproduction_matrix.svg` 생성
- `framework_lab.py`: reproduction experiment card schema, comparison policy, scope gate, artifact hygiene checklist를 deterministic framework metrics로 기록
- `analysis.py`: metrics가 없으면 actionable Korean error로 실패하고, 실행별 관측은 `artifacts/analysis-manual/latest_report.md`에 기록
- `analysis.md`: 실행마다 덮어쓰지 않는 안정적인 해석 프레임
- `reflection.md`: Korean-first 회고 질문과 artifact checklist

## 실습 흐름
1. 논문 전체가 아니라 핵심 claim 1~3개를 고른다.
2. scope control을 먼저 건다. 이번 단위에서는 real paper/code download, full benchmark, GPU training을 제외하고 reduced claim만 다룬다.
3. claim/evidence matrix를 만든다. claim, evidence type, acceptance rule, observed signal, decision을 한 행으로 묶는다.
4. baseline/reported/reproduced comparison을 분리한다. paper 표 숫자와 local reproduced 숫자를 직접 섞지 않고, 같은 protocol 비교를 primary comparison으로 둔다.
5. seed variance와 reported gap을 비교해 숫자 차이가 의미 있는지 먼저 확인한다.
6. mismatch hypothesis를 남긴다. preprocessing_alignment, seed_variance, budget_mismatch 같은 후보를 다음 실험 질문으로 바꾼다.
7. artifact hygiene checklist로 scope boundary, matrix, comparison table, variance summary, mismatch hypotheses가 모두 남았는지 확인한다.

## 실행 방법
```bash
python 07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py
python 07_frontier_labs/01_paper_reproduction_playground/framework_lab.py
python 07_frontier_labs/01_paper_reproduction_playground/analysis.py
```

## 실행 결과 예시
아래 예시는 이 저장소의 deterministic toy data로 실제 생성되는 metrics 구조를 축약한 것이다.

```text
$ python 07_frontier_labs/01_paper_reproduction_playground/scratch_lab.py
{
  "status": "runnable",
  "mode": "claim_level_reproduction_playground",
  "claim_evidence_matrix": [
    {"claim_id": "C1_adapter_efficiency", "decision": "direction_reproduced_with_small_gap"}
  ],
  "comparisons": {
    "C1_adapter_efficiency": {
      "baseline": {"accuracy": 0.842},
      "reported": {"accuracy": 0.851},
      "reproduced": {"accuracy": 0.846},
      "delta_vs_baseline": 0.004,
      "delta_vs_reported": -0.005
    }
  },
  "artifacts": {"figure": "artifacts/scratch-manual/paper_reproduction_matrix.svg"}
}

$ python 07_frontier_labs/01_paper_reproduction_playground/framework_lab.py
{
  "framework": "cpu_deterministic_reproduction_harness",
  "runtime_contract": {"network_policy": "offline_no_network_no_paper_download"},
  "comparison_policy": {"primary_comparison": "same_protocol_reproduced_baseline_vs_method"}
}

$ python 07_frontier_labs/01_paper_reproduction_playground/analysis.py
# 01 Paper Reproduction Playground 실행 관측
- scratch/framework metrics를 읽고 latest_report.md를 갱신한다.
```

생성 파일:
- `artifacts/scratch-manual/metrics.json`
- `artifacts/scratch-manual/paper_reproduction_matrix.svg`
- `artifacts/framework-manual/metrics.json`
- `artifacts/analysis-manual/latest_report.md`

## 이 단위에서 특히 볼 질문
- “논문을 재현한다”는 말은 full paper clone인가, claim-level reproduction인가?
- reduced claim으로 scope를 줄였다면 원래 paper claim 중 무엇을 더 이상 말하면 안 되는가?
- baseline, reported, reproduced 숫자를 한 표에 놓을 때 primary comparison은 무엇인가?
- reproduced result가 reported result보다 낮을 때 variance와 mismatch hypothesis를 어떻게 먼저 기록할 것인가?
- artifact hygiene가 없다면 다음 사람이 어떤 조건을 다시 추적해야 하는가?

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/02_capstone_model_building`에서는 남의 논문 claim을 따라가는 데서 한 걸음 더 나아가 내 프로젝트의 성공 기준과 비교선을 직접 설계한다. 이 단위의 claim/evidence matrix와 artifact hygiene는 capstone에서 문제 정의, baseline, eval protocol, failure analysis를 닫는 템플릿으로 이어진다.
