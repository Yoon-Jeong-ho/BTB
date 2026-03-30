# 02 Activation and Loss

## 왜 이 단위를 배우는가
activation과 loss를 구분하지 못하면 `logits -> probabilities -> error signal` 흐름이 흐릿해진다. 이 단위는 **비선형성은 activation이 만들고, 학습 목표는 loss가 만든다**는 사실을 아주 작은 숫자 예제로 몸에 익히게 한다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/activation_curves.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 ReLU / sigmoid / tanh / softmax를 직접 계산하고 toy loss를 수식 수준에서 확인한다.
2. `framework_lab.py`에서 PyTorch activation/loss API가 같은 개념을 어떻게 계산하는지 tiny tensor로 확인한다.
3. `analysis.py`로 관측치를 한국어 문장으로 정리하고, 안정적인 해석 문서와 실행별 리포트를 분리한다.

## 실행 결과 예시
```text
$ python 00_foundations/02_activation_and_loss/scratch_lab.py
{
  "relu_zero_fraction": 0.555556,
  "binary_cross_entropy": 0.251929,
  "cross_entropy": 0.162877,
  "figure_path": "artifacts/scratch-manual/activation_curves.svg"
}

$ python 00_foundations/02_activation_and_loss/framework_lab.py
{
  "row_probability_sums": [1.0, 1.0],
  "cross_entropy_loss": 0.217482,
  "binary_cross_entropy_loss": 0.359588
}
```
실행 후에는 SVG figure와 metrics JSON이 `artifacts/` 아래에 쌓여, 계산 결과를 바로 눈으로 확인할 수 있다.

## 다음 단위와의 연결
이 단위의 감각은 이후 optimizer/backprop, attention score 해석, GPU runtime 분석에서 모두 재사용된다. 특히 `logits를 바로 loss에 넣는지`, `확률로 바꾼 뒤 비교하는지`를 구분하는 습관이 중요하다.
