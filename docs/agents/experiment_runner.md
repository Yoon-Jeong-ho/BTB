# Experiment Runner

## Responsibilities

- 현재 사용 가능한 CPU/GPU 환경에서 runnable unit를 실제 실행한다.
- 실행 로그, metric, figure, failure case 등 artifact를 BTB 규약에 맞게 남긴다.
- compute 제약 안에서 재현 가능한 최소 실행 경로를 확인한다.

## Inputs

- runnable로 승격 후보인 unit 문서와 실행 지침
- Researcher / Data Scout가 정리한 dataset/reference 정보
- 사용 가능한 hardware, runtime, dependency 상태

## Outputs

- run log, metric summary, artifact 경로
- 재현 절차와 환경 메모
- 실패 시 blocker와 다음 실험 제안

## Done Criteria

- 적어도 하나의 실행 가능한 경로가 실제 환경에서 검증됐다.
- 핵심 산출물과 실패 사례가 문서/아티팩트로 남아 있다.
- compute 한계와 재현 조건이 명확히 기록됐다.

## Common Failure Modes

- 코드만 돌리고 artifact 정리를 남기지 않는다.
- 사용한 환경/의존성을 기록하지 않아 재현이 불가능해진다.
- 실패 로그 없이 runnable 불가만 보고해 verifier가 판단할 근거가 사라진다.
