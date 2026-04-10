# 03 DeepSpeed ZeRO 선행 개념

## 꼭 알고 오면 좋은 것
- DDP에서 각 rank가 모델 복제본을 들고 gradient를 동기화한다는 기본 그림
- optimizer state, gradient, parameter가 서로 다른 메모리 덩어리라는 점
- batch size, micro-batch, gradient accumulation, effective batch의 차이
- mixed precision이 memory footprint와 step 안정성에 영향을 준다는 점
- distributed collective(all-reduce, all-gather 같은 연산)가 왜 필요한지에 대한 아주 기본 감각
- GPU 메모리 부족이 단순 OOM 메시지 이상으로 training design 문제라는 관점

## 먼저 다시 보면 좋은 단위
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — 메모리 병목과 runtime 관찰 감각 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — effective batch, accumulation, logging 질문 복습
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — DDP 복제/동기화 기본 그림 복습
- [06_training_systems/02_accelerate_workflows](../02_accelerate_workflows/README.md) — framework-level launcher/config 추상화 감각 연결

## 빠른 자기 점검
- DDP에서 optimizer state까지 각 rank가 사실상 복제해서 들고 있으면 왜 메모리 한계가 빨리 오는지 설명할 수 있는가?
- optimizer state, gradient, parameter 셋 중 무엇이 Adam 계열에서 특히 큰 메모리 비용을 만들기 쉬운지 알고 있는가?
- micro-batch를 줄이거나 gradient accumulation을 늘리는 것과, state 자체를 shard하는 것은 무엇이 다른지 구분할 수 있는가?
- ZeRO Stage 1/2/3가 각각 어떤 상태를 partition하는지 순서대로 말할 수 있는가?
- memory saving이 커질수록 communication/orchestration 비용이 늘 수 있다는 trade-off를 받아들일 준비가 되어 있는가?
