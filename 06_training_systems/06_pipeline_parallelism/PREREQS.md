# 06 Pipeline Parallelism 선행 개념

## 꼭 알고 오면 좋은 것
- transformer block이나 깊은 MLP처럼 순차 레이어 스택이 어떻게 이어지는지에 대한 기본 감각
- forward / backward / activation이 메모리와 runtime에서 어떤 흔적을 남기는지에 대한 이해
- batch, microbatch, gradient accumulation이 서로 다른 운영 개념이라는 점
- DDP 기준 rank / world size / main-process 운영 규칙에 대한 기본 감각
- tensor parallel이 레이어 내부 연산을 나누는 방식이라는 점
- GPU 메모리 부족과 통신 병목이 모델 설계와 별개의 시스템 설계 문제라는 관점

## 먼저 다시 보면 좋은 단위
- [02_deep_learning/04_attention_and_transformers](../../02_deep_learning/04_attention_and_transformers/README.md) — transformer 레이어 스택과 residual 흐름 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — activation / memory / runtime 관찰 감각 복습
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — 분산 rank와 launch 기본 계약 복습
- [06_training_systems/03_deepspeed_zero](../03_deepspeed_zero/README.md) — 메모리 절감과 분산 state 관리 감각 연결
- [06_training_systems/05_tensor_parallelism](../05_tensor_parallelism/README.md) — 레이어 내부 분할과 stage 분할의 차이 예습

## 빠른 자기 점검
- data parallel, tensor parallel, pipeline parallel을 각각 한 문장으로 구분해 설명할 수 있는가?
- microbatch를 늘리는 것이 batch size를 늘리는 것과 왜 완전히 같은 말이 아닌지 이해하는가?
- 어떤 stage가 다른 stage보다 느리면 pipeline 전체 throughput이 그 stage에 묶인다는 점을 받아들일 수 있는가?
- stage boundary에서 activation을 옮긴다는 것이 실제 장치 간 통신 문제라는 점을 설명할 수 있는가?
- pipeline bubble이 warmup/cooldown에서 생기는 idle time이라는 점을 시간축으로 그려 볼 수 있는가?
