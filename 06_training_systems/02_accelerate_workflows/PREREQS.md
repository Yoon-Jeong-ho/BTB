# 02 Accelerate Workflows 선행 개념

## 꼭 알고 오면 좋은 것
- single-GPU PyTorch 학습 루프에서 model / optimizer / dataloader / scheduler가 어떻게 연결되는지
- `torchrun`, rank, world size, local rank가 왜 필요한지에 대한 아주 기본 감각
- `.to(device)` / `.cuda()` 호출이 model과 batch를 장치에 올리는 기본 방식이라는 점
- mixed precision(fp16/bf16)의 목적이 속도/메모리 절약이지만 수치 안정성 질문을 함께 만든다는 점
- gradient accumulation이 effective batch를 바꾸는 운영 도구라는 점
- distributed training에서도 train/eval logging, metric gather, checkpoint 저장 복구가 필요하다는 점

## 먼저 다시 보면 좋은 단위
- [01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — rank/world size/DDP의 최소 계약 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — device/memory/runtime 감각 복습
- [02_deep_learning/07_training_recipes_and_debugging](../../02_deep_learning/07_training_recipes_and_debugging/README.md) — mixed precision, gradient accumulation, debugging 질문 복습
- [05_advanced_nlp_llm/04_instruction_tuning_and_sft](../../05_advanced_nlp_llm/04_instruction_tuning_and_sft/README.md) — Hugging Face 생태계 training loop와 later fine-tuning 연결 감각 확보

## 빠른 자기 점검
- DDP를 직접 쓸 때 왜 launcher, rank, device placement, dataloader sharding이 반복 보일러플레이트가 되는지 설명할 수 있는가?
- `.to(device)`를 코드 곳곳에 넣는 방식이 single-GPU에서 multi-GPU/TPU 확장 시 왜 번거로워지는지 이해하는가?
- mixed precision을 "속도 옵션"으로만 보지 않고 overflow/precision support 문제와 함께 봐야 한다는 점을 받아들일 수 있는가?
- dataloader가 분산 환경에서 프로세스별로 나뉘지 않으면 어떤 중복/불일치 문제가 생길지 설명할 수 있는가?
- launcher를 단순화하는 도구가 있다고 해도 backend(DDP/DeepSpeed/FSDP) 원리를 따로 이해해야 하는 이유를 말할 수 있는가?
