# 03 Agentic Training and Eval Loops

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
앞 단위에서 capstone 문제, baseline, milestone, failure slice를 문서로 고정했다면, 이제 남는 질문은 하나다. **그 계약을 누가 어떤 순서로 실행하고, 언제 멈추고, 어떤 근거로 다음 시도를 고를 것인가?** frontier 프로젝트에서 agent를 붙인다는 말은 단순 자동 실행 버튼을 추가하는 일이 아니라, training/eval workflow 전체를 작은 반복 실험 단위로 쪼개고 각 반복마다 증거를 남기는 운영 루프를 설계하는 일에 가깝다.

이 단위는 agentic workflow를 마법 같은 자동화로 소개하지 않는다. 오히려 planner / executor / verifier / critic 역할을 분리해 두지 않으면, 실험은 빨라져도 **잘못된 metric chasing, retry storm, 근거 없는 self-approval**가 더 빨라질 수 있다는 점을 먼저 다룬다. 즉, agent를 실험 가속기가 아니라 **증거 기반 실험 운영자**로 붙이는 감각을 익히는 것이 목적이다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- agentic training/eval workflow의 역할 분리와 반복 구조를 정리한 `THEORY.md`
- 선행 개념과 자기 점검 질문을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - planner brief와 iteration plan
  - executor run log / metric / artifact bundle
  - verifier checklist와 gate verdict
  - critic triage memo, retry 정책, escalation note

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable/applied 승격 때 구현할 실습 순서다.
1. 먼저 이전 capstone에서 고정한 problem statement, baseline, acceptance gate를 agent가 읽을 수 있는 **실험 계약(experiment contract)** 으로 다시 쓴다. 목표가 모호하면 loop는 빨라지기만 하고 방향은 잃는다.
2. planner는 이번 반복에서 바꿀 변수와 바꾸지 않을 변수를 구분한다. 예를 들어 learning rate만 바꾸는지, data slice까지 바꾸는지, retry budget은 몇 번인지처럼 **탐색 범위와 종료 조건**을 먼저 적는다.
3. executor는 정해진 계약 안에서 실제 train/eval run을 수행한다. 중요한 것은 실행 자체보다도 config, seed, split, hardware, runtime, failure log를 같이 남겨 **다음 반복이 같은 근거를 다시 읽을 수 있게** 만드는 것이다.
4. verifier는 숫자를 받아 적는 역할이 아니라, 현재 run이 같은 benchmark / 같은 eval protocol / 같은 artifact contract를 만족했는지 확인한다. 즉 "점수가 올랐다"보다 먼저 **비교 가능한 run인가**를 판단한다.
5. critic는 결과를 보고 다음 행동을 고른다. retry, rollback, scope 축소, human escalation 중 무엇이 맞는지 결정하면서, improvement보다도 **왜 이 실험이 믿을 만한지 / 왜 아직 못 믿는지**를 문장으로 남긴다.
6. 마지막에는 loop trace를 정리해 다음 단위 `07_frontier_labs/04_benchmark_and_dataset_construction`으로 넘긴다. 그래야 agent가 무엇을 최적화하고 있는지, benchmark 자체가 흔들리고 있는지까지 연결해서 볼 수 있다.

## 이 단위에서 특히 볼 질문
- agentic training/eval loop는 단순 job automation과 무엇이 다르고, 왜 iteration contract가 먼저 필요한가?
- planner / executor / verifier / critic를 한 agent에 몰아넣으면 어떤 self-justification 문제가 생기는가?
- retry를 많이 하는 것이 항상 좋은가, 아니면 variance / budget / contamination 때문에 오히려 해석을 망칠 수 있는가?
- verifier는 metric checker인가, 아니면 artifact와 protocol 정합성을 확인하는 gatekeeper인가?
- critic가 제안한 다음 행동이 실제 evidence에 기반한 것인지, 단순 직감인지 어떻게 구분할 수 있는가?
- 언제 agent loop를 멈추고 사람에게 escalation해야 하는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 07_frontier_labs/03_agentic_training_and_eval_loops/scratch_lab.py
{
  "status": "sample",
  "loop_id": "agentic-train-eval-v1",
  "experiment_contract": {
    "goal": "baseline 대비 Recall@10 개선",
    "frozen_constraints": ["same eval split", "same primary metric", "same budget tier"],
    "retry_budget": 3
  },
  "planner": {
    "iteration": 2,
    "change_set": ["lr=3e-5", "hard_negative_ratio=0.3"],
    "stop_if": ["artifact missing", "delta < variance band after 2 retries"]
  },
  "executor": {
    "train_status": "finished",
    "runtime_minutes": 78,
    "hardware": "1xA100",
    "artifact_paths": ["artifacts/run_002/train_log.jsonl", "artifacts/run_002/eval_metrics.json"]
  },
  "verifier": {
    "protocol_match": true,
    "artifact_complete": true,
    "baseline_comparable": true,
    "warnings": ["seed count=1, variance 해석 제한"]
  },
  "critic": {
    "verdict": "retry_with_narrower_change",
    "reason": "metric 개선은 있으나 verifier warning 때문에 일반화 주장 불가",
    "next_focus": ["seed 2개 추가", "hard negative sampling 영향 분리"]
  }
}

$ python 07_frontier_labs/03_agentic_training_and_eval_loops/analysis.py
{
  "status": "sample",
  "history_shape": [4, 8],
  "observations": [
    "planner가 두 변수 이상을 동시에 바꾼 run에서 해석력이 급격히 떨어짐",
    "verifier가 artifact 누락을 잡은 run은 critic triage 전에 rollback됨",
    "retry 수가 늘수록 improvement보다 benchmark contamination 위험 관찰 필요"
  ],
  "escalation_rules": [
    "same failure 2회 반복 시 human review",
    "benchmark drift 의심 시 loop 중단 후 dataset contract 재검토"
  ]
}
```

핵심은 agent가 많이 일했다는 사실이 아니라, **각 반복이 어떤 계약 아래 실행되었고 어떤 증거가 남았으며 누가 어떤 기준으로 다음 행동을 고른 것인지**를 읽을 수 있어야 한다는 점이다.

## 다음 단위와의 연결
다음 단위 `07_frontier_labs/04_benchmark_and_dataset_construction`에서는 이 loop가 무엇을 최적화하고 무엇을 gate로 삼는지 더 근본적으로 묻는다. agentic loop가 좋아 보여도 benchmark와 dataset contract가 약하면, 결국 자동화된 속도로 잘못된 신호를 최적화하게 된다. 그래서 이 단위에서 **iteration trace, verifier gate, escalation 기준**을 먼저 잡아 두면, 다음 단위에서는 benchmark/dataset 자체를 어떻게 설계해야 loop가 정직해지는지로 자연스럽게 이어진다.
