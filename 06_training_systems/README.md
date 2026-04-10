# 06 Training Systems

이 트랙은 `05_advanced_nlp_llm` 이후에 놓이는 **distributed / large-model systems** 구간이다. 모델 objective와 post-training 흐름을 이해한 뒤, 이제는 `그 모델을 실제 하드웨어 위에서 어떻게 안정적으로 학습시키는가`를 다룬다.

즉 `02~05`에서 얻은 모델 감각을 `06`에서 시스템 감각으로 바꾸고, 이후 `07_frontier_labs`에서 더 큰 실험·재현·캡스톤을 수행할 수 있도록 분산 실행, 병렬화, checkpoint, failure recovery를 정리하는 역할을 맡는다.

## 단위 구성

| Unit | Status | Focus |
| --- | --- | --- |
| [01_torchrun_and_ddp_basics](01_torchrun_and_ddp_basics/README.md) | outlined | torchrun/DDP로 다중 프로세스 학습을 시작하는 최소 계약을 익힌다. |
| [02_accelerate_workflows](02_accelerate_workflows/README.md) | outlined | Hugging Face Accelerate로 실험 스크립트를 이식성 있게 운영하는 방법을 본다. |
| [03_deepspeed_zero](03_deepspeed_zero/README.md) | outlined | optimizer/state sharding으로 메모리 병목을 줄이는 ZeRO 단계를 정리한다. |
| [04_fsdp_checkpointing_and_offload](04_fsdp_checkpointing_and_offload/README.md) | planned | FSDP와 activation/checkpoint/offload 조합으로 큰 모델을 다루는 법을 배운다. |
| [05_tensor_parallelism](05_tensor_parallelism/README.md) | planned | 레이어 내부 연산을 여러 장치로 나누는 tensor parallel의 구조를 이해한다. |
| [06_pipeline_parallelism](06_pipeline_parallelism/README.md) | planned | 모델 층을 스테이지로 쪼개 pipeline bubble과 schedule trade-off를 본다. |
| [07_data_parallel_grad_accumulation](07_data_parallel_grad_accumulation/README.md) | planned | effective batch를 키우기 위한 data parallel과 grad accumulation 조합을 정리한다. |
| [08_hybrid_parallel_topologies](08_hybrid_parallel_topologies/README.md) | planned | data/tensor/pipeline/FSDP를 섞는 실제 대형 모델 topology를 비교한다. |
| [09_profiling_monitoring_and_failure_recovery](09_profiling_monitoring_and_failure_recovery/README.md) | planned | 병목 측정, 로그 관찰, 장애 복구 runbook을 학습한다. |

## 이 트랙에 포함되는 것

- torchrun/DDP, Accelerate, DeepSpeed ZeRO, FSDP, checkpoint/offload 같은 단일-다중 노드 훈련 시스템 기본기
- tensor/pipeline/data parallel, grad accumulation, hybrid topology 설계
- profiling, monitoring, 장애 복구, 재시작 전략

## 이 트랙에서 아직 다루지 않는 것

- language objective, instruction tuning, alignment policy 설계는 `05_advanced_nlp_llm`에서 다룬다.
- 논문 재현과 open-ended research 운영은 `07_frontier_labs`에서 다룬다.
- 기초 neural architecture 자체의 원리는 `02_deep_learning`에서 먼저 다룬다.
