# 07 Training Recipes and Debugging 선행 개념

## 꼭 알고 오면 좋은 것
- loss, metric, gradient가 서로 다른 신호라는 점
- optimizer step / epoch / batch / global step 같은 기본 학습 루프 용어
- train split과 validation split을 나누는 이유
- overfit와 underfit를 아주 기본적인 학습 곡선 수준에서 구분하는 감각
- tensor shape와 target 형식이 loss 함수와 맞아야 한다는 점
- GPU 메모리, mixed precision, gradient accumulation이 학습 설정에 영향을 준다는 기본 이해

## 먼저 다시 보면 좋은 단위
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — gradient가 update로 이어지는 흐름 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — 메모리/throughput/runtime 관찰 감각 복습
- [01_ml/03_model_selection_and_interpretation](../../01_ml/03_model_selection_and_interpretation/README.md) — validation 기반 비교와 해석 습관 복습
- [02_deep_learning/04_attention_and_transformers](../../02_deep_learning/04_attention_and_transformers/README.md) — later transformer/LLM 연결을 위한 모델 배경 복습
- [02_deep_learning/06_generative_models_vae_gan](../../02_deep_learning/06_generative_models_vae_gan/README.md) — 불안정한 학습 곡선과 failure pattern을 보는 문제의식 연결

## 빠른 자기 점검
- learning rate를 너무 크게 잡으면 왜 단순히 "빨라지는 것"이 아니라 발산으로 이어질 수 있는가?
- train loss가 내려가는데 validation metric이 나빠질 때, 가장 먼저 어떤 가능성을 의심해야 하는가?
- batch size를 키우면 메모리 외에 gradient noise와 effective batch가 바뀐다는 말을 이해하는가?
- loss 함수와 target 형식이 맞지 않으면 NaN 또는 이상한 학습 로그가 생길 수 있다는 점을 설명할 수 있는가?
- single-batch overfit이나 tiny-subset replay가 왜 디버깅 첫 단계로 유용한지 말할 수 있는가?
