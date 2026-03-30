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

## Common Confusion
- activation과 loss를 둘 다 “출력 함수”처럼 기억하는 실수
- sigmoid/softmax로 확률을 만든 뒤, logits를 기대하는 loss에 다시 넣는 실수
- loss 값이 작다고 해서 activation이 항상 좋아졌다고 단정하는 실수
- ReLU의 0 출력이 “계산 실패”라고 오해하는 실수

## 실행에서 확인할 포인트
- `artifacts/scratch-manual/activation_curves.svg`에서 ReLU / sigmoid / tanh 곡선이 어떻게 다르게 생겼는지 본다.
- scratch와 framework 결과 모두에서 softmax 행 합이 1인지 확인한다.
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
