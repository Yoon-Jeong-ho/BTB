# 02 Activation and Loss 이론 노트

## activation이 필요한 이유
- 선형 층(linear layer)만 여러 번 쌓으면 결국 큰 하나의 선형 변환으로 합쳐진다.
- activation은 중간에 **꺾이는 규칙(non-linearity)** 를 넣어, 모델이 더 복잡한 decision boundary를 표현하게 만든다.
- 그래서 activation은 "표현력"과 연결되고, loss는 "무엇을 잘해야 하는가"와 연결된다.

## 대표 activation을 어떻게 읽을까
- **ReLU**: 음수는 0으로 자르고 양수는 그대로 둔다. sparse activation 감각을 만들기 쉽다.
- **sigmoid**: 값을 0~1 사이로 눌러 binary probability처럼 읽기 좋다. BCE(binary cross entropy)와 자주 연결된다.
- **tanh**: 값을 -1~1 사이로 눌러 centered activation 감각을 준다.
- **softmax**: 여러 class logits를 확률 분포로 바꿔 각 행의 합이 1이 되게 만든다.

## loss는 무엇을 하나
- loss는 예측과 정답 사이의 차이를 **하나의 scalar** 로 압축한다.
- 이 scalar가 backpropagation의 출발점이 되므로, activation이 만든 표현을 실제 학습 신호로 연결하는 다리 역할을 한다.
- binary 분류에서는 `BCEWithLogitsLoss`, multi-class 분류에서는 `CrossEntropyLoss`가 흔하다.

## logits / probability / loss 연결
- logits는 아직 정규화되지 않은 점수다.
- softmax나 sigmoid는 logits를 probability처럼 읽을 수 있게 변환한다.
- 하지만 PyTorch 손실 함수 다수는 수치 안정성 때문에 **probability가 아니라 logits를 직접 받는 버전** 을 제공한다.
- 따라서 `softmax를 먼저 하고 CrossEntropyLoss를 또 적용하는 실수`를 피해야 한다.

### 왜 logits를 직접 받으면 더 안정적인가

probability로 바꾼 뒤 loss를 계산하면 식이 직관적이다.

```text
binary cross entropy = -y log(p) - (1-y) log(1-p)
```

하지만 logit이 아주 크거나 작으면 sigmoid/softmax가 float 안에서 거의 `0.0` 또는 `1.0`으로 포화된다. 그 다음에 `log(p)` 또는 `log(1-p)`를 계산하면 `log(0)`이 되어 `inf`, `nan`, 또는 지나치게 큰 값이 생길 수 있다.

예를 들어 logit이 `1000`이고 정답이 `0`이면 모델은 매우 자신 있게 틀린 것이다.

```text
sigmoid(1000) ≈ 1.0
naive BCE = -log(1 - 1.0) = -log(0) = inf 또는 계산 실패
```

반면 `BCEWithLogitsLoss`는 sigmoid를 먼저 만든 뒤 log를 취하지 않고, 아래처럼 같은 의미의 안정식으로 계산한다.

```text
max(x, 0) - x*y + log(1 + exp(-abs(x)))
```

이 식은 `exp(1000)`처럼 터지는 값을 직접 만들지 않는다. `x=1000, y=0`이면 loss는 대략 `1000`이라는 유한한 값으로 남는다. 즉 **logits가 문제가 없다**기보다, **logits를 받는 loss 함수가 sigmoid/softmax와 log를 한 번에 묶어 overflow/underflow가 덜 나는 식으로 계산한다**가 정확하다.

PyTorch의 `binary_cross_entropy`처럼 probability를 받는 함수가 내부 clamp로 `inf`를 피하는 경우도 있다. 하지만 이미 sigmoid가 `1.0`으로 포화된 뒤라, “logit 1000으로 틀림”과 “훨씬 더 큰 logit으로 틀림”의 차이는 확률 공간에서 사라진다. 이 단위의 `numeric_stability_demo`에서 naive probability BCE가 `100.0`으로 제한되고, logits 기반 BCE가 `1000.0`을 유지하는 이유가 여기에 있다.

`CrossEntropyLoss`도 같은 이유로 `softmax → log → NLLLoss`를 따로 하지 않고, 내부적으로 `log_softmax` 계열의 안정 계산을 사용한다. 그래서 사람이 볼 때는 probability를 출력해도 되지만, 학습 loss에는 logits를 직접 넣는 습관이 안전하다.

## Common Confusion
- activation과 loss를 둘 다 “출력 함수”처럼 기억하는 실수
- sigmoid/softmax로 확률을 만든 뒤, logits를 기대하는 loss에 다시 넣는 실수
- loss 값이 작다고 해서 activation이 항상 좋아졌다고 단정하는 실수
- ReLU의 0 출력이 “계산 실패”라고 오해하는 실수

## 실행에서 확인할 포인트
- `artifacts/scratch-manual/activation_curves.svg`에서 ReLU / sigmoid / tanh 곡선이 어떻게 다르게 생겼는지 본다. 왼쪽은 ReLU까지 포함한 공통 y축이라 sigmoid가 눌려 보일 수 있고, 오른쪽 확대 패널에서 sigmoid의 S-curve 곡률을 확인한다.
- `framework_lab.py` 실행 결과의 `activation_rows`를 한 줄씩 읽어, 같은 입력이 ReLU / sigmoid / tanh를 지나며 어떤 값으로 바뀌는지 비교한다.
- scratch와 framework 결과 모두에서 softmax 행 합이 1인지 확인한다.
- `numeric_stability_demo`에서 `sigmoid(1000)` 이후 naive BCE와 logits 기반 BCE가 어떻게 달라지는지 확인한다.
- BCE와 cross entropy가 각각 어떤 정답 형식(binary / class index)을 기대하는지 비교한다.

## 실행 결과 예시
```text
scratch metrics
- relu_zero_fraction: 0.555556
- softmax_argmax: 0
- binary_cross_entropy: 0.251929
- cross_entropy: 0.162877

framework metrics
- row_probability_sums: [1.0, 1.0]
- cross_entropy_loss: 0.217482
- binary_cross_entropy_loss: 0.359588
```
이 숫자는 “activation은 값의 모양을 바꾸고, loss는 그 결과를 학습 가능한 scalar로 줄인다”는 흐름을 아주 작은 예제로 보여준다.
