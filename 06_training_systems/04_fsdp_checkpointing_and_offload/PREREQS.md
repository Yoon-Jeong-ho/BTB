# 04 FSDP, Checkpointing, and Offload 선행 개념

## 꼭 알고 오면 좋은 것
- DDP와 ZeRO에서 rank별로 상태를 복제하거나 shard한다는 기본 그림
- parameter, gradient, optimizer state, activation이 서로 다른 메모리 덩어리라는 점
- all-gather, reduce-scatter 같은 collective communication이 언제 필요해지는지에 대한 기초 감각
- mixed precision이 메모리 절감과 수치 안정성 질문을 함께 만든다는 점
- checkpoint 저장/복구가 단순 파일 저장이 아니라 training state contract라는 점
- GPU 메모리 부족 문제를 batch 축소만이 아니라 runtime/orchestration 문제로 보는 관점

## 먼저 다시 보면 좋은 단위
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — 메모리 병목과 device/runtime 관찰 감각 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — activation checkpointing, mixed precision, failure triage 질문 복습
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — rank/world size/DDP 통신 기본 좌표계 복습
- [06_training_systems/03_deepspeed_zero](../03_deepspeed_zero/README.md) — state partitioning intuition과 memory-vs-communication trade-off 기준선 복습

## 빠른 자기 점검
- DDP나 단순 data parallel에서는 왜 parameter full replica가 메모리 병목이 되기 쉬운지 설명할 수 있는가?
- activation memory와 optimizer state memory를 서로 다른 병목으로 구분해 말할 수 있는가?
- activation checkpointing이 왜 메모리를 줄이지만 step time을 늘릴 수 있는지 이해하는가?
- CPU offload가 GPU 메모리를 줄이는 대신 host-device transfer 비용을 만든다는 점을 받아들일 준비가 되어 있는가?
- full state dict와 sharded state dict가 save/load/debug/export 관점에서 왜 다른 운영 계약을 가진다고 보는지 설명할 수 있는가?
