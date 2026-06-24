# 01 Vision-Language-Action Grounding 분석 노트

이 문서는 반복 실행에도 유지되는 해석 프레임이다. 실제 실행별 수치는 `artifacts/analysis-manual/latest_report.md`에 남긴다.

## 해석 프레임

- VLA는 image-text understanding을 action selection으로 확장한다.
- `action_accuracy`는 지시와 장면에 맞는 action token을 골랐는지 본다.
- `safety_gate_accuracy`는 위험 상태에서 실행을 막거나 stop action을 선택하는지 본다.
- 실제 VLA 실험으로 확장할 때는 success rate, trajectory error, intervention count, safety violation을 함께 남긴다.

## 관련 이론

- [THEORY.md](./THEORY.md)
- [PREREQS.md](./PREREQS.md)
