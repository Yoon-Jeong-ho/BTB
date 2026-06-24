# 01 Vision-Language-Action Grounding 분석 노트

이 문서는 반복 실행에도 유지되는 해석 프레임이다. 실제 실행별 수치는 `artifacts/analysis-manual/latest_report.md`에 남긴다.

## 해석 프레임

- VLA는 image-text understanding을 action selection으로 확장한다.
- `action_accuracy`는 지시와 장면에 맞는 action token을 골랐는지 본다.
- `safety_gate_accuracy`는 위험 상태에서 실행을 막거나 stop action을 선택하는지 본다.
- 실제 VLA 실험으로 확장할 때는 success rate, trajectory error, intervention count, safety violation을 함께 남긴다.

## 실패 probe 기록법

실행 결과를 읽을 때는 아래 네 칸으로 실패를 나눈다.

| Probe | 해석 질문 |
| --- | --- |
| wrong action but safe | safety gate가 위험은 막았지만 목표 행동 학습이 부족한가? |
| right action but unsafe | action token은 맞아도 safety constraint가 빠져 실제 실행이 위험한가? |
| ambiguous instruction | 바로 행동하지 말고 stop/clarify를 내야 하는 지시인가? |
| observation noise | 시각 feature가 흔들려 action score와 safety score가 같이 불안정해졌는가? |

## 관련 이론

- [THEORY.md](./THEORY.md)
- [PREREQS.md](./PREREQS.md)
