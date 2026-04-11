# 03 Agentic Training and Eval Loops Reflection

## 실행 전 점검

- 이번 loop의 experiment contract는 무엇을 목표로 하고 무엇을 고정하는가?
- baseline, primary metric, split, budget tier, artifact schema가 같은 문서에 묶여 있는가?
- retry budget, stop rule, escalation rule이 실행 전에 적혀 있는가?

## 역할 분리 점검

### planner

- planner가 제안한 change set은 한두 개 변수로 제한되어 있는가?
- planner가 expected evidence를 먼저 적었는가?
- planner가 자신의 plan을 직접 approve하지 않도록 verifier gate가 분리되어 있는가?

### executor

- executor는 승인된 change set만 실행했는가?
- seed, config_hash, runtime, metric_json, artifact_manifest, failure_slice_report가 evidence bundle에 남았는가?
- executor가 metric을 해석하거나 claim을 만들지 않았는가?

### verifier

- verifier가 protocol match를 metric 개선보다 먼저 확인했는가?
- artifact completeness가 깨진 run을 통과시키지 않았는가?
- baseline comparability와 benchmark drift를 별도 gate로 남겼는가?

### critic

- critic verdict가 evidence_refs를 인용하는가?
- retry / rollback / stop / escalation 중 무엇을 골랐고, 왜 그 선택이 retry budget과 stop rule에 맞는가?
- critic가 "좋아 보인다"가 아니라 "아직 주장할 수 없는 것"을 먼저 적었는가?

## 실습 후 질문

1. experiment contract 없이 agentic loop를 돌리면 어떤 protocol mismatch가 생기기 쉬운가?
2. planner / executor / verifier / critic를 분리하지 않을 때 self-approval은 어떤 문장으로 나타나는가?
3. retry budget은 언제 도움이 되고, 언제 retry storm을 막기 위한 stop signal이 되는가?
4. verifier가 artifact completeness를 확인하지 않으면 어떤 evidence bundle 필드가 빠질 수 있는가?
5. benchmark drift가 관측되면 왜 더 많은 자동 retry보다 escalation rule이 먼저인가?
6. 이번 simulation의 final gate가 `needs_human_review`인 이유를 evidence bundle 관점에서 설명할 수 있는가?

## 나의 loop contract 초안

- experiment contract id:
- primary metric:
- frozen constraints:
- retry budget:
- stop rule:
- escalation rule:
- required evidence bundle:
- planner output:
- executor output:
- verifier gate:
- critic triage:

## 다음 단위로 넘길 메모

다음 `benchmark_and_dataset_construction` 단위에서 다시 확인해야 할 benchmark/dataset 질문:

- drift probe가 어떤 data slice에서 경고를 냈는가?
- 현재 metric이 실제 목표 claim을 측정하는가?
- artifact completeness가 깨진 run은 dataset/schema 문제였는가, executor hygiene 문제였는가?
- agentic loop가 최적화한 신호가 benchmark gaming으로 이어질 위험은 없는가?
