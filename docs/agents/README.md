# Agent Roles for BTB Curriculum Expansion

## 목적

이 디렉터리는 BTB 커리큘럼 확장 Phase 1에서 사람이 따라야 할 역할 분담을 정리한다. 자동화 규약이 아니라, `planned -> outlined -> runnable` 승격을 안정적으로 진행하기 위한 한국어 우선 운영 문서다.

## Phase 1 Workflow

1. **Program Director**가 lane 범위와 우선순위를 정하고 owner를 배정한다.
2. **Curriculum Architect**가 트랙 구조, numbering, prerequisite, status progression을 고정한다.
3. **Theory Writer**가 `README` / `THEORY` / `PREREQS` / reflection skeleton 초안을 만든다.
4. **Researcher / Data Scout**가 논문, 데이터셋, benchmark, reference implementation을 수집한다.
5. **Experiment Runner**가 현재 사용 가능한 CPU/GPU에서 runnable unit를 실행하고 artifact를 남긴다.
6. **Critic / Verifier**가 문서 품질, 링크, 산출물, claim-evidence 정합성을 점검한다.

## 상태 승격 원칙

- `planned`: 트랙/유닛의 역할과 범위만 합의된 상태
- `outlined`: 문서 skeleton, prerequisite, 참고자료, 실행 계획이 정리된 상태
- `runnable`: 실제 실행 가능한 unit와 artifact 규약이 확인된 상태

Program Director는 승격 결정을 내리기 전에 Curriculum Architect, Theory Writer, Experiment Runner, Critic / Verifier의 산출물을 모두 확인해야 한다.

## 역할 간 handoff

- Program Director → lane brief, owner, 목표 상태
- Curriculum Architect → 구조/선수지식/상태 표준
- Theory Writer → 독자가 읽을 skeleton 문서
- Researcher / Data Scout → 근거 자료와 reference 묶음
- Experiment Runner → run log, artifact, 재현 메모
- Critic / Verifier → merge 가능 여부와 수정 요구사항

## 운영 원칙

- 문서는 한국어 우선을 유지하고 필요한 영어 technical term만 병기한다.
- runnable 승격은 claim이 아니라 evidence 기준으로 판단한다.
- 새 트랙을 추가해도 현재 `00 -> 09` ladder와 상태 모델을 깨지 않는다.
- 이 역할 문서는 expansion workflow를 돕기 위한 운영 가이드이며, 자동 실행 계약으로 간주하지 않는다.
