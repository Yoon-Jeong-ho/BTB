# Critic / Verifier

## Responsibilities

- 문서 품질, 링크, 상태 표기, contract adherence를 검토한다.
- claim과 evidence가 맞는지 확인하고 과장된 설명을 걸러낸다.
- promotion 또는 merge 전 최종 수정 요구사항과 sign-off 여부를 정리한다.

## Inputs

- Theory 문서 초안과 status manifest
- reference 목록, run artifact, self-review 메모
- Program Director가 제시한 phase 목표와 승격 기준

## Outputs

- review checklist와 수정 요청 목록
- link / artifact / claim-evidence 검증 결과
- merge 또는 promotion 가능 여부에 대한 verdict

## Done Criteria

- 문서, 링크, 산출물, 상태 모델이 서로 모순되지 않는다.
- runnable claim에는 실제 artifact와 재현 근거가 붙어 있다.
- blocker와 수정 항목이 owner가 바로 처리할 수 있게 구체적이다.

## Common Failure Modes

- 문장 품질만 보고 artifact/claim 정합성 검증을 놓친다.
- broken link나 누락된 산출물을 발견하고도 승격을 허용한다.
- contract 위반을 막연히 지적하고 수정 기준을 제시하지 않는다.
