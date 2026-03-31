# 00 Foundations

이 구간은 모든 상위 트랙 전에 공통 기초를 맞추는 진입 계단이다.

## 여기서 다루는 것

- tensor shape, broadcasting, indexing
- activation, loss, logits, gradient
- optimizer와 backpropagation 감각
- normalization, regularization, training dynamics 기초
- GPU memory, runtime, debugging 기초
- tokenization, embedding, attention의 최소 배경

## 단위 구성

1. `01_tensor_shapes/` — 텐서 shape, broadcasting, matmul, batch 차원을 먼저 읽는 훈련
2. `02_activation_and_loss/` — activation이 값을 어떻게 꺾고, loss가 오차를 어떻게 하나의 숫자로 압축하는지 실험
3. `03_gradients_and_backpropagation/` — gradient, chain rule, finite-difference check, autograd/backprop를 숫자와 그림으로 확인
4. `04_regularization_and_normalization/` — 입력 scale 정리, LayerNorm, dropout, weight decay가 학습 안정성과 weight growth를 어떻게 바꾸는지 관측
5. `05_gpu_memory_runtime/` — GPU/CPU runtime, dtype, training/inference 차이를 숫자로 관측

## 읽는 방법

한글 설명을 먼저 읽고, 필요한 technical term만 영어로 연결해서 이해한다.
각 단위는 `README -> THEORY -> scratch_lab/framework_lab -> analysis -> reflection` 순서로 보는 것을 권장한다.
