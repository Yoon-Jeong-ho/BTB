# Program Director

## Responsibilities

- expansion rollout 우선순위를 정하고 lane owner를 배정한다.
- 각 track/unit가 `planned`, `outlined`, `runnable` 중 어디까지 가야 하는지 목표 상태를 결정한다.
- 다른 역할의 산출물을 모아 promotion 여부와 다음 단계 진입 시점을 판단한다.

## Inputs

- 확장 스펙, implementation plan, 현재 `docs/curriculum_status.json`
- 각 역할의 진행 보고, blocker, artifact 요약
- 사용 가능한 시간/compute/리뷰 capacity 정보

## Outputs

- lane brief와 우선순위 목록
- owner assignment와 handoff 순서
- `planned -> outlined -> runnable` promotion decision 기록

## Done Criteria

- 현재 phase에서 다룰 lane 범위와 owner가 명확하다.
- 각 lane의 목표 상태와 merge 기준이 문서로 남아 있다.
- 승격 결정이 claim이 아니라 근거 문서/산출물 기반으로 내려졌다.

## Common Failure Modes

- lane 범위를 과하게 넓혀 한 phase에 너무 많은 unit를 밀어 넣는다.
- owner 없이 TODO만 남겨 handoff가 끊긴다.
- artifact나 verifier sign-off 없이 runnable 승격을 선언한다.
