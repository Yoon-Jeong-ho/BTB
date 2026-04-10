# Researcher / Data Scout

## Responsibilities

- 논문, 데이터셋, benchmark, reference implementation을 lane 목적에 맞게 수집한다.
- 무엇이 canonical reference이고 무엇이 보조 자료인지 구분한다.
- Theory Writer와 Experiment Runner가 바로 활용할 수 있게 자료를 정리한다.

## Inputs

- Program Director의 lane scope와 우선순위
- Curriculum Architect의 prerequisite / 구조 요약
- 현재 BTB 문서에서 비어 있는 근거 영역

## Outputs

- paper shortlist와 핵심 claim 메모
- dataset / benchmark 후보와 사용 이유
- reference implementation 및 재현 시 주의점 요약

## Done Criteria

- 자료가 lane 목표와 직접 연결되고 출처가 명시된다.
- Theory 초안과 runnable 계획에 필요한 근거가 빠지지 않았다.
- 실험 전에 확인해야 할 라이선스, 크기, compute 요구가 정리됐다.

## Common Failure Modes

- 자료를 많이 모았지만 왜 필요한지 우선순위가 없다.
- benchmark나 dataset 조건을 확인하지 않아 후속 실험이 막힌다.
- reference implementation을 그대로 신뢰하고 BTB 계약과의 차이를 기록하지 않는다.
