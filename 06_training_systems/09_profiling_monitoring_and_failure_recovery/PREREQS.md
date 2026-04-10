# 09 Profiling, Monitoring, and Failure Recovery 선행 개념

## 꼭 알고 오면 좋은 것
- training step, throughput, step time이 서로 어떻게 연결되는지에 대한 기본 감각
- loss / grad norm / optimizer step 로그를 읽는 최소한의 학습 디버깅 직관
- GPU memory의 allocated / reserved / peak 개념과 runtime 관찰 기본기
- DDP/FSDP/ZeRO/pipeline 같은 distributed runtime에서 rank 간 대기와 통신 병목이 생길 수 있다는 이해
- checkpoint가 단순 model save를 넘어 restart state 계약이라는 점
- "느리다 / 멈췄다 / 품질이 무너졌다"를 서로 다른 failure symptom으로 나눠 볼 준비

## 먼저 다시 보면 좋은 단위
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — GPU memory/runtime 관찰 기본기 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — loss, grad norm, instability, 디버깅 감각 복습
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — rank, world size, main-process 운영 규칙 복습
- [06_training_systems/04_fsdp_checkpointing_and_offload](../04_fsdp_checkpointing_and_offload/README.md) — sharded state, checkpoint, offload, resume 관점 연결
- [06_training_systems/06_pipeline_parallelism](../06_pipeline_parallelism/README.md) — time-axis scheduling과 communication wait 감각 복습

## 빠른 자기 점검
- throughput 저하와 loss divergence를 서로 다른 관찰 문제로 나눠 설명할 수 있는가?
- allocated / reserved / peak memory를 구분해서 말할 수 있는가?
- distributed run에서 한 rank가 느려지면 왜 전체 step time이 늘 수 있는지 설명할 수 있는가?
- checkpoint가 있어도 resume가 실패할 수 있는 이유를 두세 가지 이상 떠올릴 수 있는가?
- OOM, hang, divergence를 봤을 때 각각 다른 첫 질문을 던져야 한다는 점을 받아들일 수 있는가?
