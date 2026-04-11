# 03 Agentic Training and Eval Loops 분석

## Stable interpretation

Agentic training/eval loops are an operating layer around experiments, not a promise that agents should run training jobs forever. The central unit is an experiment contract: a frozen baseline, metric, split, budget, evidence bundle, retry budget, stop rule, and escalation rule.

## Korean-first reading

- experiment contract가 먼저 고정되어야 planner가 바꿀 변수와 고정할 변수를 구분할 수 있다.
- planner / executor / verifier / critic 역할 분리는 self-approval를 줄이는 최소 안전장치다.
- verifier는 metric checker가 아니라 protocol match, artifact completeness, baseline comparability, evidence bundle 완성도를 확인하는 gatekeeper다.
- critic는 verifier gate를 통과한 evidence field를 인용해 retry / rollback / stop / escalation 중 하나를 골라야 한다.
- retry budget은 탐색 범위를 제한하는 장치다. protocol mismatch나 artifact gap이 반복되면 retry가 아니라 rollback 또는 escalation이 맞다.
- benchmark drift가 관측되면 loop를 더 빠르게 돌리는 것이 아니라 benchmark/dataset construction contract를 다시 봐야 한다.

## Observed run

`analysis.py`는 `artifacts/scratch-manual/metrics.json`과 `artifacts/framework-manual/metrics.json`을 읽어 실행별 관측 보고서를 `artifacts/analysis-manual/latest_report.md`에 쓰고, 기계가 읽기 쉬운 요약을 `artifacts/analysis-manual/observed_summary.json`에 쓴다.

이 stable 문서는 실행 숫자를 고정하지 않는다. 대신 어떤 실행이든 다음 질문으로 읽게 만든다.

1. experiment contract가 baseline / metric / split / retry budget / stop rule을 충분히 고정했는가?
2. planner가 한 iteration에서 바꾼 change set은 해석 가능한가?
3. executor artifact가 다음 사람이 재현하고 검토할 만큼 충분한가?
4. verifier gate가 protocol match와 artifact completeness를 metric claim보다 먼저 확인했는가?
5. critic recommendation은 evidence bundle을 인용하는가, 아니면 plausible한 추측인가?
6. benchmark drift가 보일 때 stop/escalation rule이 자동 retry보다 우선했는가?

## Failure signals to watch

- protocol match 없이 점수만 좋아지는 run
- artifact completeness가 깨졌는데 critic가 improvement를 주장하는 run
- evidence bundle에 config_hash, seed, split, verifier_gate, critic_triage가 빠진 run
- 같은 verifier failure를 retry budget으로 덮는 run
- benchmark drift나 contamination signal을 무시하고 더 많은 iteration을 수행하는 run

좋은 agentic loop는 "계속 돌았다"가 아니라, **왜 지금 멈췄고 어떤 evidence 때문에 사람 검토가 필요한지**를 설명할 수 있어야 한다.
