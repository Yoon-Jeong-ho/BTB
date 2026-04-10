# 01 Perceptron and MLP 이론 노트

## 핵심 개념
- **perceptron**은 입력 `x`에 가중치 `w`와 bias `b`를 적용한 뒤, `w·x + b`의 부호로 클래스를 나누는 가장 단순한 neural classifier다.
- 이 규칙이 만드는 decision boundary는 2차원에서는 직선, 더 높은 차원에서는 hyperplane이다.
- 어떤 데이터가 **linear separable** 하다는 말은, 직선(또는 hyperplane) 하나로 두 클래스를 완전히 나눌 수 있다는 뜻이다.
- **single neuron**은 선형 결합 뒤 activation 하나를 거친 구조이고, **MLP(multi-layer perceptron)** 는 hidden layer와 nonlinearity를 추가해 입력을 한 번 더 재표현한다.
- 핵심은 **linear + linear는 다시 linear** 라는 점이다. activation이 빠지면 층을 늘려도 표현력은 거의 늘지 않는다.

## perceptron decision rule을 어떻게 읽을까
- `w`는 어떤 feature 방향을 중요하게 볼지 정한다.
- `b`는 경계를 원점에서 얼마나 밀어낼지 정한다.
- `w·x + b`가 0보다 크면 한 클래스, 작으면 다른 클래스로 보낸다.
- 따라서 perceptron을 이해하는 핵심은 "모델이 feature 공간에서 어디에 선을 긋는가"를 읽는 것이다.

## 왜 XOR가 중요한가
- XOR는 대각선에 있는 두 점을 같은 클래스로 묶어야 한다.
- 직선 하나는 공간을 반평면 두 개로만 나누므로, 대각선 패턴을 동시에 만족시키기 어렵다.
- 그래서 XOR 실패는 single neuron의 학습 부족이라기보다 **표현력 부족** 을 보여 주는 고전 예제다.
- tiny MLP는 hidden layer에서 중간 표현을 만든 뒤, 그 표현 위에서 다시 선형 분류를 수행해 XOR를 해결할 수 있다.

## tiny MLP를 너무 거창하게 생각하지 말자
- 이 단위의 tiny MLP는 hidden layer 하나뿐이다.
- parameter 수가 조금 늘어날 뿐인데도 표현력은 크게 달라진다.
- 즉 큰 모델로 가기 전에, **작은 MLP baseline이 어디까지 해내는지** 먼저 보는 습관이 중요하다.

## 실행 결과 예시
```text
$ python 02_deep_learning/01_perceptron_and_mlp/scratch_lab.py
{
  "linear_dataset_accuracy": 1.0,
  "xor_best_accuracy": 0.75,
  "xor_failure_reason": "직선 하나로는 XOR의 대각선 패턴을 동시에 나눌 수 없다."
}

$ python 02_deep_learning/01_perceptron_and_mlp/framework_lab.py
{
  "single_neuron_xor_loss": 0.693147,
  "single_neuron_xor_accuracy": 0.5,
  "tiny_mlp_xor_loss": 0.001726,
  "tiny_mlp_xor_accuracy": 1.0,
  "xor_accuracy_gain": 0.5
}
```
이 숫자는 "single neuron은 XOR에서 거의 랜덤 수준에 머물고, tiny MLP는 같은 작은 toy data에서 분명히 나아진다"는 메시지를 보여 준다.

## Common Confusion
- perceptron과 logistic regression을 완전히 같은 모델이라고 생각하는 실수
- hidden layer만 넣으면 activation 없이도 복잡한 경계를 만들 수 있다고 오해하는 실수
- accuracy가 낮은 원인을 항상 optimizer 문제로만 보고, linear separability와 표현력 한계를 먼저 보지 않는 실수
- tiny MLP baseline을 "너무 약해서 의미 없다"고 넘겨 버리는 실수

## 다음 단계로 이어지는 질문
- image classification에서는 왜 fully connected MLP만으로 공간 구조를 놓치기 쉬울까?
- sequence 문제에서는 hidden state나 attention이 hidden layer 역할을 어떻게 확장할까?
- 더 깊은 네트워크가 생길수록 표현력과 최적화 난이도는 어떤 식으로 같이 커질까?
