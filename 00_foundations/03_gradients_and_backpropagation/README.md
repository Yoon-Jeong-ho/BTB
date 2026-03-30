# 03 Gradients and Backpropagation

## 왜 이 단위를 배우는가
activation과 loss를 이해한 다음에는, **오차가 각 파라미터까지 어떻게 거꾸로 전달되는지**를 볼 차례다. 이 단위는 아주 작은 선형 예제와 tiny PyTorch network를 통해 gradient, chain rule, backpropagation을 숫자와 그림으로 확인하게 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/loss_curve.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 scalar 선형 모델의 forward, loss, analytic gradient, finite-difference gradient check를 직접 계산한다.
2. `framework_lab.py`에서 PyTorch autograd가 tiny network의 gradient를 어떻게 채우는지 관측하고 한 번의 optimizer step으로 loss 감소를 확인한다.
3. `analysis.py`로 수치 관측을 한국어 문장으로 정리하고, 안정적인 해석 문서와 실행별 리포트를 분리한다.

## 실행 결과 예시
```text
$ python 00_foundations/03_gradients_and_backpropagation/scratch_lab.py
{
  "loss": 0.125,
  "grad_w": 0.75,
  "finite_diff_grad_w": 0.75,
  "updated_loss": 0.056953,
  "figure_path": "artifacts/scratch-manual/loss_curve.svg"
}

$ python 00_foundations/03_gradients_and_backpropagation/framework_lab.py
{
  "loss_before_step": 0.40695,
  "loss_after_step": 0.119054,
  "first_layer_weight_grad_norm": 1.080176,
  "total_grad_norm": 1.39978
}
```
실행 후에는 JSON metrics와 SVG figure가 `artifacts/` 아래에 생겨, gradient 방향과 loss 감소를 바로 확인할 수 있다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: chain rule, local gradient, finite-difference gradient check의 개념을 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 gradient/loss 값을 읽는다.

## 다음 단위와의 연결
이 단위에서 얻는 감각은 optimizer, learning rate, attention parameter update, GPU training cost 해석으로 바로 이어진다. 특히 “어느 항이 어느 항에 곱해져 gradient가 만들어지는가”를 말로 설명할 수 있어야 이후 실험이 빨라진다.
