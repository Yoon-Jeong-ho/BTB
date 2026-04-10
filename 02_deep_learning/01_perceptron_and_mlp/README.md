# 01 Perceptron and MLP

> Status: outlined

## 왜 이 단위를 배우는가
`00_foundations`에서 activation, loss, gradient를 따로 봤다면 이제는 그 조각들이 **가장 작은 neural classifier** 안에서 어떻게 한 덩어리로 움직이는지 봐야 한다. perceptron은 "가중합에 문턱을 적용해 한쪽 클래스로 보낼지 말지 결정하는 규칙"을 보여 주고, MLP는 그 규칙 하나로는 부족할 때 hidden layer와 nonlinearity가 왜 필요한지 설명해 준다. 이 단위는 이후 CNN/RNN/Transformer를 보기 전에 **single neuron으로 가능한 것과 불가능한 것**을 먼저 분리하는 역할을 한다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`, `THEORY.md`, `PREREQS.md`
- 단위 목적, 핵심 용어, 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어올 자리 `artifacts/.gitkeep`
- runnable 승격 때 채울 `decision boundary`, `toy metrics`, `baseline comparison` 출력 계약

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. `scratch_lab.py`(예정)에서 perceptron의 `sign(w·x + b)` 규칙으로 2차원 점을 어떻게 둘로 가르는지 본다.
2. 같은 toy 데이터에 대해 선형 분리 가능 예제와 XOR 같은 비선형 예제를 나란히 두고, single neuron의 한계를 비교한다.
3. `framework_lab.py`(예정)에서 hidden layer 1개짜리 tiny MLP를 추가해 nonlinearity가 decision boundary를 어떻게 굽히는지 관찰한다.
4. `analysis.py` / `analysis.md`(예정)에서 "가장 작은 neural baseline을 왜 먼저 세우는가"를 한국어 문장으로 정리한다.

## 이 단위에서 특히 볼 질문
- perceptron의 decision rule은 입력을 어떤 기준으로 양쪽 클래스로 나누는가?
- "직선 하나로 나눌 수 있다 / 없다"는 말이 정확히 무엇을 뜻하는가?
- single neuron과 logistic-regression류 baseline은 어디까지 비슷하고, 어디서 MLP가 필요해지는가?
- hidden layer가 늘어나면 무엇이 달라지고, nonlinearity가 빠지면 왜 다시 선형 모델처럼 보이는가?
- 실전에서 가장 작은 neural baseline은 어느 수준에서 출발하는 것이 좋은가?

## 실행 결과 예시
아래는 **아직 실행을 완료했다는 뜻이 아니라**, 이후 `scratch_lab.py` / `framework_lab.py`가 추가되었을 때 기대하는 출력 형식의 예시다.

```json
{
  "dataset": "toy_points",
  "decision_rule": "sign(w·x + b)",
  "linear_separable_accuracy": 1.0,
  "xor_accuracy": 0.5,
  "figure_path": "artifacts/scratch-manual/decision_regions.svg"
}
```

```json
{
  "model": "tiny_mlp",
  "hidden_units": 4,
  "activation": "tanh",
  "linear_separable_accuracy": 1.0,
  "xor_accuracy": 1.0,
  "notes": "sample output shape only"
}
```

실제 runnable 단계에서는 seed, 초기화, 데이터 포인트 배치에 따라 숫자가 달라질 수 있지만, **선형 분리 가능 데이터와 XOR류 데이터를 나란히 비교하는 출력 구조**는 유지하는 것이 핵심이다.

## 다음 단위와의 연결
이 감각이 있어야 `02_cnn_and_image_classification`에서 "왜 flatten만으로는 공간 구조를 잘 못 잡는가"를 이해하기 쉽고, 뒤의 sequence/attention 단위에서도 "선형층 + nonlinearity를 여러 번 쌓아 표현력을 키운다"는 공통 구조가 더 선명하게 보인다.
