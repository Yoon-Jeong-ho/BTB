# 01 Perceptron and MLP 이론 노트

## 핵심 개념
- **perceptron**은 입력 벡터 `x`에 가중치 `w`와 bias `b`를 적용한 뒤, 가중합 `w·x + b`의 부호나 문턱값으로 클래스를 결정하는 가장 단순한 분류 규칙이다.
- 이 규칙이 만드는 decision boundary는 2차원에서는 직선, 더 높은 차원에서는 hyperplane이다.
- 어떤 데이터가 **linear separable** 하다는 말은, 직선(또는 hyperplane) 하나로 두 클래스를 완전히 나눌 수 있다는 뜻이다.
- **single neuron**은 선형 결합 뒤 activation 하나를 거친 구조이고, **MLP(multi-layer perceptron)** 는 이런 뉴런을 hidden layer까지 포함해 여러 층으로 쌓은 모델이다.
- hidden layer에 **nonlinearity** 가 들어가야 여러 선형 조각을 조합한 더 복잡한 decision boundary를 만들 수 있다.
- 그래서 perceptron은 "가장 작은 규칙 기반 neural classifier", MLP는 "가장 작은 범용 neural baseline"으로 이해할 수 있다.

## 수식 / 직관
- perceptron의 기본 점수는 `z = w^T x + b` 로 쓴다.
- 고전적 perceptron decision rule은 보통 `ŷ = 1 if z >= 0 else 0` 처럼 문턱을 적용한다.
- sigmoid를 쓰면 `p = σ(z)` 로 0~1 사이의 값으로 바꿔 probability처럼 읽을 수 있다. 이때는 hard threshold보다 학습 가능한 differentiable classifier 쪽에 더 가깝다.
- 중요한 점은 **linear + linear만 쌓으면 결국 다시 linear** 라는 사실이다.
  - `W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)`
  - 즉 hidden layer를 넣어도 activation이 없으면 표현력이 늘지 않는다.
- hidden unit 여러 개는 "서로 다른 직선 여러 개"를 만든 뒤, 그 조합으로 더 복잡한 영역을 근사한다고 보면 된다.
- XOR가 대표적인 예시다. 점 네 개를 직선 하나로는 나눌 수 없지만, hidden layer와 nonlinearity를 넣으면 두 개 이상의 선형 조각을 조합해 해결할 수 있다.

## perceptron decision rule을 어떻게 읽을까
- `w`는 각 입력 feature를 얼마나 중요하게 볼지 정한다.
- `b`는 경계를 원점에서 얼마나 밀어낼지 정한다.
- `w·x + b`가 크면 한 클래스 쪽, 작으면 다른 클래스 쪽으로 기운다.
- 따라서 perceptron을 이해하는 핵심은 "모델이 feature 공간에서 어느 방향으로 선을 긋는가"를 읽는 것이다.

## linear separability가 왜 중요한가
- 선형 분리 가능 데이터에서는 single neuron 하나로도 꽤 강한 baseline이 된다.
- 반대로 선형 분리 불가능 데이터에서는 아무리 learning rate나 epoch를 조정해도 직선 하나의 한계가 남는다.
- 이 구분을 먼저 이해해야, 이후 더 큰 모델을 쓸 때도 "최적화가 실패한 것인지, 표현력이 부족한 것인지"를 분리해 볼 수 있다.

## single neuron에서 MLP로 넘어가는 이유
- single neuron은 입력을 한 번에 한쪽/다른쪽으로 나누는 규칙 한 개라고 생각할 수 있다.
- MLP는 hidden layer를 통해 중간 표현을 다시 만들고, 그 표현 위에서 다시 분류한다.
- 그래서 MLP는 "원래 입력 공간에서 바로 직선을 긋는 모델"이 아니라, "입력을 한 번 재표현한 뒤 분류하는 모델"이다.
- 실전에서는 거대한 모델보다 먼저 작은 MLP baseline을 두면, 데이터가 정말 복잡한지 아니면 단순 baseline으로도 충분한지 빠르게 판단할 수 있다.

## Common Confusion
- perceptron과 logistic regression을 완전히 같은 모델이라고 생각하는 실수
- hidden layer만 추가하면 activation 없이도 복잡한 경계를 만들 수 있다고 오해하는 실수
- linear separable이 아니라는 말을 "신경망으로도 못 푼다"로 과장하는 실수
- 작은 MLP baseline을 "너무 약해서 의미 없는 실험"이라고 넘겨 버리는 실수
- accuracy가 낮은 원인을 항상 optimizer 문제로만 보고, 표현력 한계를 먼저 의심하지 않는 실수

## 이 단위에서 관찰할 것
- 2차원 toy point를 그렸을 때 직선 하나로 깨끗하게 나뉘는지 먼저 본다.
- 선형 분리 가능 예제와 XOR 예제를 single neuron으로 돌렸을 때 어떤 실패 패턴이 나타나는지 비교한다.
- hidden unit 수와 activation 종류를 바꿨을 때 decision boundary가 얼마나 꺾이거나 여러 조각으로 나뉘는지 본다.
- 가장 작은 neural baseline이 파라미터 수를 크게 늘리지 않고도 어떤 문제를 해결 범위에 넣는지 확인한다.

## 다음 단계로 이어지는 질문
- image classification에서는 왜 fully connected MLP만으로 공간 구조를 놓치기 쉬울까?
- sequence 문제에서는 hidden state나 attention이 hidden layer 역할을 어떻게 확장할까?
- 더 깊은 네트워크가 생길수록 표현력과 최적화 난이도는 어떤 식으로 같이 커질까?
