# 03 Gradients and Backpropagation 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측 요약은 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 gradient와 backpropagation을 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- gradient는 loss를 줄이는 방향 정보를 담고 있으므로, 부호와 크기를 함께 읽어야 한다.
- backpropagation은 마지막 오차 신호를 앞단 local gradient와 곱해 각 파라미터로 전달한다.
- scratch의 finite-difference gradient check는 analytic gradient 구현이 맞는지 검증하고, framework의 autograd는 같은 개념을 일반 tensor graph로 확장한다.
- `artifacts/scratch-manual/loss_curve.svg`는 현재 weight와 gradient step 이후 weight가 loss 곡선에서 어떻게 이동하는지 보여준다.

## 확인 질문
- analytic gradient와 finite-difference gradient가 거의 같다는 사실은 무엇을 보장하는가?
- backpropagation에서 `d(loss)/d(prediction)`이 앞단 gradient와 어떻게 결합되는가?
- 이번 실행에서 loss 감소와 gradient norm은 `artifacts/analysis-manual/latest_report.md`에 어떻게 기록되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): chain rule, finite-difference check, autograd 흐름을 다시 확인한다.
