# 08 Hybrid Parallel Topologies 선행 개념

## 꼭 알고 오면 좋은 것
- data parallel, tensor parallel, pipeline parallel, FSDP/ZeRO가 각각 무엇을 나누는지에 대한 큰 그림
- global batch, microbatch, gradient accumulation이 서로 다른 운영 축이라는 점
- large Transformer/LLM에서 parameter state, activation, intra-layer matmul이 서로 다른 병목이라는 감각
- all-reduce, all-gather, reduce-scatter, send/recv가 어떤 상황에서 등장하는지에 대한 기본 직관
- node 내부 빠른 링크와 node 간 느린 링크가 topology 배치에 큰 영향을 준다는 점
- checkpoint 저장/복구가 단순 파일 저장이 아니라 topology-aware 운영 계약이 될 수 있다는 점

## 먼저 다시 보면 좋은 단위
- [06_training_systems/04_fsdp_checkpointing_and_offload](../04_fsdp_checkpointing_and_offload/README.md) — state sharding, offload, checkpoint contract 복습
- [06_training_systems/05_tensor_parallelism](../05_tensor_parallelism/README.md) — intra-layer split과 collective communication 감각 복습
- [06_training_systems/06_pipeline_parallelism](../06_pipeline_parallelism/README.md) — stage / microbatch / bubble / activation transfer 복습
- [06_training_systems/07_data_parallel_grad_accumulation](../07_data_parallel_grad_accumulation/README.md) — effective batch와 synchronization cadence 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — 메모리 병목과 runtime 관찰 감각 복습

## 빠른 자기 점검
- data parallel, tensor parallel, pipeline parallel, FSDP를 각각 "무엇을 나누는가" 기준으로 한 문장씩 설명할 수 있는가?
- 어떤 통신은 node 내부에 가두고, 어떤 축은 node 간으로 보내는 편이 유리한지 하드웨어 링크 관점에서 말할 수 있는가?
- 모델이 메모리에 안 들어가는 문제와 throughput이 부족한 문제를 서로 다른 topology 질문으로 분리할 수 있는가?
- global batch를 맞추기 위해 data parallel만 늘리는 것과 grad accumulation / pipeline microbatch를 조정하는 것이 왜 다른 계약인지 이해하는가?
- hybrid topology가 fit만 되면 끝나는 것이 아니라 checkpoint, restart, profiling, failure isolation까지 영향을 준다는 점을 받아들일 준비가 되어 있는가?
