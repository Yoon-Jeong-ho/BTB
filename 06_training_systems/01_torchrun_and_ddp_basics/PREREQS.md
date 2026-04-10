# 01 Torchrun and DDP Basics 선행 개념

## 꼭 알고 오면 좋은 것
- optimizer step, backward, gradient가 학습 루프에서 어떤 역할을 하는지에 대한 기본 감각
- GPU 하나에서 모델/optimizer/batch가 어떻게 연결되는지에 대한 이해
- batch size와 step 수가 학습 로그 해석에 영향을 준다는 점
- Python 프로세스가 각각 독립된 메모리/상태를 가진다는 아주 기본적인 감각
- rank별 로그가 섞이면 디버깅이 어려워질 수 있다는 운영 직관
- distributed 학습이 "더 많은 GPU"만이 아니라 "더 많은 process"를 다루는 문제라는 점

## 먼저 다시 보면 좋은 단위
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — backward와 gradient update가 어디서 생기는지 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — GPU 장치/메모리/runtime 관찰 감각 복습
- [01_ml/04_large_scale_tabular](../../01_ml/04_large_scale_tabular/README.md) — 배치 크기와 큰 데이터셋 처리 관점 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — batch, effective batch, logging discipline, failure triage 감각 연결

## 빠른 자기 점검
- single-process 학습에서는 왜 rank 개념이 필요 없고, distributed launch에서는 왜 갑자기 필요해지는지 설명할 수 있는가?
- `world_size`, `rank`, `local_rank`를 각각 한 문장으로 구분해 말할 수 있는가?
- 각 rank가 local batch를 따로 처리한 뒤 gradient를 맞추면 optimizer step 결과가 왜 비슷하게 유지되는지 직관적으로 설명할 수 있는가?
- 왜 `torchrun` 없이 손으로 여러 process를 띄우는 것보다 launcher 계약이 중요한지 이해하는가?
- 왜 logging/checkpoint를 모든 rank가 동시에 수행하지 않고 main process 중심으로 정리하는지 설명할 수 있는가?
