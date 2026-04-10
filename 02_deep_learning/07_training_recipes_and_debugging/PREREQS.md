# 07 학습 레시피와 디버깅 선행 개념

## 먼저 알고 오면 좋은 것
- loss, metric, gradient가 각각 다른 신호라는 점
- batch / epoch / optimizer step / global step 같은 기본 학습 루프 용어
- train/validation split을 나누는 이유와 기본적인 overfit/underfit 해석
- learning rate가 너무 크면 발산할 수 있고, 너무 작으면 underfit처럼 보일 수 있다는 감각
- label misalignment, shape mismatch, NaN/Inf 입력이 data bug로 이어질 수 있다는 점

## 다시 보면 좋은 이전 단위
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — gradient가 실제 update로 이어지는 흐름 복습
- [00_foundations/04_regularization_and_normalization](../../00_foundations/04_regularization_and_normalization/README.md) — weight decay, regularization, 안정성 직관 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — batch size와 runtime/메모리 관찰 감각 복습
- [01_ml/03_model_selection_and_interpretation](../../01_ml/03_model_selection_and_interpretation/README.md) — validation 기반 해석 습관 복습

## 빠른 자기 점검
- learning rate를 크게 올렸을 때 왜 단순히 “더 빨리 학습”이 아니라 divergence가 생길 수 있는가?
- batch size를 바꾸면 throughput 외에 gradient noise와 fit 속도 해석이 왜 달라지는가?
- train loss는 잘 내려가는데 validation loss만 이상하게 커진다면, overfit와 data bug를 어떤 순서로 구분하겠는가?
- single-batch overfit sanity check가 실패하면 모델/optimizer/data 중 무엇을 먼저 보겠는가?
