# 00 Foundations

이 구간은 BTB 전체에서 가장 먼저 보는 **딥러닝 공통 기초 트랙**이다. tensor shape를 읽는 법부터 activation, gradient, regularization, runtime까지를 한 줄의 숫자 흐름으로 연결하는 데 목적이 있다.

## 언제 먼저 보면 좋은가

- 딥러닝 내부가 아직 흐릿할 때
- activation, loss, backprop, optimizer가 따로 노는 느낌일 때
- attention을 보기 전에 shape / mask / runtime 감각을 먼저 잡고 싶을 때

## foundations 내부 추천 순서

1. [01_tensor_shapes](01_tensor_shapes/README.md) — shape, broadcasting, matmul, batch 차원을 먼저 읽는 훈련
2. [02_activation_and_loss](02_activation_and_loss/README.md) — activation이 값을 어떻게 꺾고, loss가 오차를 어떻게 하나의 숫자로 압축하는지 실험
3. [03_gradients_and_backpropagation](03_gradients_and_backpropagation/README.md) — gradient, chain rule, finite-difference check, autograd/backprop를 숫자와 그림으로 확인
4. [04_regularization_and_normalization](04_regularization_and_normalization/README.md) — 입력 scale 정리, LayerNorm, dropout, weight decay가 학습 안정성과 weight growth를 어떻게 바꾸는지 관측
5. [05_gpu_memory_runtime](05_gpu_memory_runtime/README.md) — GPU/CPU runtime, dtype, training/inference 차이를 숫자로 관측

## 여기서 다루는 것

- tensor shape, broadcasting, indexing
- activation, loss, logits, gradient
- optimizer와 backpropagation 감각
- normalization, regularization, training dynamics 기초
- GPU memory, runtime, debugging 기초
- tokenization, embedding, attention의 최소 배경

## 한 unit를 읽는 기본 순서

각 unit는 `README -> THEORY -> PREREQS -> scratch_lab/framework_lab -> analysis -> reflection` 순서로 보는 것을 권장한다.

특히 foundations에서는 아래 질문을 계속 붙잡는다.

1. 입력 shape와 출력 shape는 무엇인가?
2. 어느 단계에서 숫자의 의미가 바뀌는가?
3. loss가 어떤 실수를 강하게 벌점으로 주는가?
4. gradient나 normalization이 학습 안정성에 어떤 영향을 주는가?
5. runtime과 memory는 어디서 늘어나는가?

## 다음 단계로 어떻게 이어지나

- 기본 루트: foundations를 끝낸 뒤 [01_ml](../01_ml/README.md) 로 가서 실험 discipline을 붙인다.
- 딥러닝 코어 압축 루트: foundations를 끝낸 뒤 [02_nlp_bridge](../02_nlp_bridge/README.md) 로 바로 가서 embedding과 attention을 잇는다.

한글 설명을 먼저 읽고, 필요한 technical term만 영어로 연결해서 이해한다.
