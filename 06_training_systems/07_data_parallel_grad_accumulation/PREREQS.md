# 07 Data Parallel + Grad Accumulation 선행 개념

## 꼭 알고 오면 좋은 것
- backward, optimizer step, gradient buffer가 학습 루프에서 어떤 역할을 하는지에 대한 기본 감각
- local batch와 batch size가 GPU memory pressure에 직접 연결된다는 이해
- DDP에서 rank마다 다른 mini-batch shard를 처리하고 gradient를 맞춘다는 직관
- global batch size와 learning-rate 해석이 서로 연결된다는 기본 관점
- throughput, step latency, GPU utilization이 서로 다른 운영 지표라는 점
- loss normalization, gradient clipping, scheduler step 타이밍이 구현 세부지만 결과 해석에 큰 영향을 준다는 점

## 먼저 다시 보면 좋은 단위
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — gradient 누적과 optimizer step 기본기 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — batch 크기와 메모리/runtime 병목 감각 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — effective batch, gradient clipping, logging discipline 감각 연결
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — rank/world size/DDP synchronization 기본 계약 복습
- [06_training_systems/06_pipeline_parallelism](../06_pipeline_parallelism/README.md) — microbatch와 step cadence를 다른 병렬화 축과 구분하는 감각 연결

## 빠른 자기 점검
- local batch, global batch, effective batch를 각각 한 문장으로 구분해 말할 수 있는가?
- local batch를 키우면 왜 memory pressure가 바로 올라가고, accumulation을 늘리면 왜 optimizer step cadence가 바뀌는지 설명할 수 있는가?
- DDP에서 각 rank가 다른 데이터를 본 뒤 gradient를 맞춘다는 직관을 이미 가지고 있는가?
- accumulation을 쓸 때 loss normalization과 gradient clipping 시점을 무시하면 왜 해석이 틀어질 수 있는지 받아들일 준비가 되어 있는가?
- throughput 개선, memory fit, optimization stability가 같은 목표가 아니라 서로 trade-off 관계일 수 있다는 점을 이해하는가?
