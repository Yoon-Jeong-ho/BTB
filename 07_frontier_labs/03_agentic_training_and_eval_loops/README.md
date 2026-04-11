# 03 Agentic Training and Eval Loops

> Status: runnable
>
> CPU-safe deterministic simulation으로 agentic training/eval loop의 experiment contract, planner / executor / verifier / critic 역할 분리, retry budget, stop rule, escalation rule, protocol match, artifact completeness, evidence bundle, benchmark drift를 관찰한다. 실제 학습, GPU, 외부 서비스, 네트워크 호출은 사용하지 않는다.

## 왜 이 단위를 배우는가

앞 단위에서 capstone 문제, baseline, milestone, failure slice를 문서로 고정했다면, 이제 남는 질문은 하나다. **그 계약을 누가 어떤 순서로 실행하고, 언제 멈추고, 어떤 근거로 다음 시도를 고를 것인가?** frontier 프로젝트에서 agent를 붙인다는 말은 단순 자동 실행 버튼을 추가하는 일이 아니라, training/eval workflow 전체를 작은 반복 실험 단위로 쪼개고 각 반복마다 증거를 남기는 운영 루프를 설계하는 일에 가깝다.

이 단위는 agentic workflow를 마법 같은 자동화로 소개하지 않는다. 오히려 planner / executor / verifier / critic 역할을 분리해 두지 않으면, 실험은 빨라져도 **잘못된 metric chasing, retry storm, 근거 없는 self-approval**가 더 빨라질 수 있다는 점을 먼저 다룬다. 즉, agent를 실험 가속기가 아니라 **증거 기반 실험 운영자**로 붙이는 감각을 익히는 것이 목적이다.

## 이번 단위에서 남길 것

- `lesson.yaml` — runnable 상태, CPU-safe / deterministic 계약, 핵심 개념과 산출물 목록
- `scratch_lab.py` — 네 번의 agentic training/eval iteration을 deterministic하게 시뮬레이션하고 metrics / jsonl trace / SVG를 생성
- `framework_lab.py` — 역할 분리, retry policy, stop/escalation rules, evidence bundle, benchmark drift gate를 실행 가능한 framework contract로 생성
- `analysis.py` — scratch/framework metrics를 읽어 observed report와 JSON summary 생성
- `analysis.md` — stable interpretation: 실행 결과를 어떻게 읽어야 하는지 고정한 분석 문서
- `reflection.md` — 실습 후 planner/executor/verifier/critic 설계를 점검하는 Korean-first worksheet
- `artifacts/.gitkeep` — 생성물 위치를 보존하는 placeholder

## 실행 방법

```bash
python3 07_frontier_labs/03_agentic_training_and_eval_loops/scratch_lab.py
python3 07_frontier_labs/03_agentic_training_and_eval_loops/framework_lab.py
python3 07_frontier_labs/03_agentic_training_and_eval_loops/analysis.py
```

생성되는 주요 artifact:

```text
07_frontier_labs/03_agentic_training_and_eval_loops/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   ├── iteration_trace.jsonl
│   └── agentic_loop_trace.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    ├── latest_report.md
    └── observed_summary.json
```

## 실행 결과 예시

`scratch_lab.py`는 다음과 같은 결정 흐름을 만든다.

```json
{
  "status": "runnable",
  "loop_id": "agentic-train-eval-v1",
  "experiment_contract": {
    "task": "agentic_retrieval_eval",
    "retry_budget": 3,
    "frozen_constraints": ["same_eval_split", "same_metric_definition", "same_preprocessing_pipeline"]
  },
  "role_sequence": ["planner", "executor", "verifier", "critic"],
  "final_decision": {
    "action": "escalate_to_human",
    "reasons": ["benchmark_drift", "long_tail_slice_regression", "low_information_retry_budget"]
  }
}
```

핵심은 agent가 많이 일했다는 사실이 아니라, **각 반복이 어떤 experiment contract 아래 실행되었고 어떤 evidence bundle이 남았으며 누가 어떤 기준으로 retry / rollback / stop / escalation을 골랐는지**를 읽을 수 있어야 한다는 점이다.

## 관찰 포인트

### 1. experiment contract가 먼저다

agentic loop는 job automation이 아니다. 실행 전에 baseline, metric, frozen constraints, retry budget, stop rule, escalation rule을 고정해야 한다. 이 계약이 없으면 더 많은 run은 더 많은 evidence가 아니라 더 많은 혼란이 된다.

### 2. 역할 분리는 self-approval 방지 장치다

- planner는 이번 iteration에서 무엇을 바꿀지 정한다.
- executor는 승인된 change set만 실행하고 config, seed, runtime, artifact manifest를 남긴다.
- verifier는 metric 개선보다 먼저 protocol match와 artifact completeness를 확인한다.
- critic는 verifier gate를 통과한 evidence를 근거로 retry / rollback / stop / escalation을 고른다.

한 역할이 계획, 실행, 승인, 다음 행동을 모두 담당하면 self-approval과 metric chasing이 빨라진다.

### 3. retry budget은 탐색을 돕지만 retry storm을 막아야 한다

retry는 protocol match와 artifact completeness가 만족될 때만 의미가 있다. verifier failure가 반복되거나 improvement가 variance band 안에 있거나 benchmark drift가 threshold를 넘으면 더 돌리는 대신 멈춰야 한다.

### 4. benchmark drift는 자동화 루프의 중단 신호다

이 unit의 deterministic trace는 마지막 iteration에서 long-tail slice regression과 drift probe warning을 보여 준다. 이때 올바른 action은 더 많은 자동 retry가 아니라 benchmark/dataset contract review로 escalation하는 것이다.

## 다음 단위와의 연결

다음 단위 `07_frontier_labs/04_benchmark_and_dataset_construction`에서는 이 loop가 무엇을 최적화하고 무엇을 gate로 삼는지 더 근본적으로 묻는다. agentic loop가 좋아 보여도 benchmark와 dataset contract가 약하면, 결국 자동화된 속도로 잘못된 신호를 최적화하게 된다. 그래서 이 단위에서 **iteration trace, verifier gate, evidence bundle, stop/escalation 기준**을 먼저 잡아 두면, 다음 단위에서는 benchmark/dataset 자체를 어떻게 설계해야 loop가 정직해지는지로 자연스럽게 이어진다.
