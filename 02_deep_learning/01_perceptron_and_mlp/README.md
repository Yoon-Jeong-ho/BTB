# 01 Perceptron and MLP

> Status: runnable

## 왜 이 단위를 배우는가
perceptron은 **가중합에 문턱을 적용해 분류를 결정하는 가장 작은 neural rule** 이고, tiny MLP는 그 rule 하나로 안 되는 패턴에서 hidden layer와 nonlinearity가 왜 필요한지 보여 주는 가장 작은 확장이다. 이 단위는 "직선 하나로 충분한가?"와 "표현력이 더 필요한가?"를 먼저 분리하게 만들어, 이후 CNN/RNN/Transformer를 볼 때도 baseline 감각을 잃지 않게 한다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/decision_regions.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 perceptron decision rule `predict=1 if w·x + b >= 0 else 0`를 직접 적용해, 선형 분리 가능 데이터는 직선 하나로 풀리고 XOR는 single neuron으로 못 푼다는 사실을 본다.
2. 같은 toy 문제를 `framework_lab.py`에서 tiny PyTorch model로 다시 관측해, single neuron과 tiny MLP의 XOR 성능 차이를 확인한다.
3. `analysis.py`로 숫자를 한국어 문장으로 정리하고, 안정적 해석 문서와 실행별 리포트를 분리한다.

## 실행 결과 예시
```text
$ python 02_deep_learning/01_perceptron_and_mlp/scratch_lab.py
{
  "decision_rule": "predict=1 if w·x + b >= 0 else 0",
  "linear_dataset_accuracy": 1.0,
  "xor_best_accuracy": 0.75,
  "figure_path": "artifacts/scratch-manual/decision_regions.svg"
}

$ python 02_deep_learning/01_perceptron_and_mlp/framework_lab.py
{
  "backend": "pytorch",
  "device": "cpu",
  "single_neuron_linear_accuracy": 1.0,
  "single_neuron_xor_accuracy": 0.5,
  "tiny_mlp_xor_accuracy": 1.0,
  "xor_accuracy_gain": 0.5
}

$ python 02_deep_learning/01_perceptron_and_mlp/analysis.py
# 01 Perceptron and MLP 실행 관측
...
```
실행 후에는 JSON metrics와 SVG figure가 `artifacts/` 아래에 생기고, `analysis.md`는 안정적인 해석 프레임만 유지한 채 `artifacts/analysis-manual/latest_report.md`에 이번 실행 숫자를 따로 기록한다.
PyTorch가 없는 환경이라면 `framework_lab.py`는 `backend: python-fallback` 형태의 최소 관측을 남기고, PyTorch가 있을 때는 위 예시처럼 실제 tiny model 비교를 기록한다.

## 문서를 읽을 때 볼 포인트
- `README.md`: 무엇을 실행하고 어떤 산출물을 남기는지 먼저 본다.
- `THEORY.md`: perceptron decision rule, linear separability, XOR failure를 개념적으로 정리한다.
- `analysis.md`: 숫자가 바뀌어도 유지되는 해석 프레임을 본다.
- `artifacts/analysis-manual/latest_report.md`: 이번 실행에서 실제로 나온 accuracy와 비교 문장을 읽는다.

## 초보자에게 특히 중요한 질문
- perceptron의 decision rule은 입력을 어떤 기준으로 양쪽 클래스로 나누는가?
- XOR 실패를 learning rate 탓이 아니라 표현력 한계라고 말하려면 무엇을 봐야 하는가?
- hidden layer와 nonlinearity가 추가되면 왜 같은 2차원 toy 문제에서도 가능한 경계가 달라지는가?

## 다음 단위와의 연결
이 감각이 있어야 `02_cnn_and_image_classification`에서 "왜 fully connected layer만으로는 공간 구조를 놓치기 쉬운가"를 이해하기 쉽다. 또한 sequence/attention 단위에서도 "선형층 + nonlinearity를 여러 번 쌓아 표현력을 만든다"는 공통 구조를 더 선명하게 읽게 된다.
