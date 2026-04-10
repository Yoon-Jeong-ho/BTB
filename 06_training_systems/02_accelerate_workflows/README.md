# 02 Accelerate Workflows

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`torchrun`/DDP를 이해한 뒤에도 실제 실험 코드는 여전히 자주 복잡해진다. 단일 GPU에서 잘 돌던 PyTorch loop를 multi-GPU, mixed precision, gradient accumulation, DeepSpeed/FSDP 준비 단계로 옮기기 시작하면 `.to(device)`, rank-aware logging, launcher 인자, dataloader sharding, backward 호출 방식이 코드 곳곳에 스며든다. Hugging Face Accelerate는 이 복잡도를 "모두 없애는 마법"이 아니라, **같은 학습 루프를 여러 실행 환경으로 옮길 때 반복되는 보일러플레이트를 줄이는 얇은 적응 계층** 으로 보는 편이 정확하다.

이 단위는 Accelerate를 "Trainer를 대신하는 거대한 프레임워크"로 오해하지 않고, **PyTorch training loop 위에 붙는 실행/장치 추상화** 로 이해하게 만든다. 그래야 다음 단위의 `06_training_systems/03_deepspeed_zero`, `04_fsdp_checkpointing_and_offload`에서 Accelerate가 어디까지 도와주고 어디부터는 backend 자체를 읽어야 하는지 경계를 잡을 수 있다.

## 이번 단위에서 남길 것
- outline 상태의 안내 문서 `README.md`
- Accelerate abstraction, device placement, mixed precision, launcher intuition을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - `Accelerator` state / device / distributed_type 관찰 메모
  - `prepare()` 전후 모델·optimizer·dataloader 변화 요약
  - mixed precision / gradient accumulation 설정 비교 표
  - `accelerate config` / `accelerate launch` workflow 체크리스트

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. single-GPU PyTorch 학습 루프를 하나 고정하고, 여기에 `.cuda()`/`.to(device)`/직접 `loss.backward()` 호출이 어디 있는지 표시한다.
2. `Accelerator()`를 도입한 뒤 `accelerator.device`, `accelerator.prepare(...)`, `accelerator.backward(...)`로 바꿔 보며 **코드에서 무엇이 사라지고 무엇은 그대로 남는지** 비교한다.
3. dataloader가 프로세스별로 어떻게 나뉘고, 모델/optimizer/scheduler가 어떤 wrapper를 거치는지 `accelerator.state`, `distributed_type`, `num_processes` 같은 관찰 포인트로 정리한다.
4. `mixed_precision="fp16"` 또는 `"bf16"` 같은 설정이 들어가면 어떤 부분이 자동으로 단순화되고, 반대로 overflow·수치 안정성·hardware support 같은 문제는 여전히 사용자가 읽어야 한다는 점을 확인한다.
5. `accelerate config`, `accelerate test`, `accelerate launch` 흐름을 통해 launcher가 rank/world-size/bootstrap 세부값을 얼마나 대신 다뤄 주는지 본다.
6. 마지막에는 Accelerate가 편하게 만드는 부분과, 실제 backend(DDP/DeepSpeed/FSDP) 이해가 여전히 필요한 부분을 분리해서 정리하며 다음 단위로 넘긴다.

## 이 단위에서 특히 볼 질문
- Accelerate는 단순 편의 래퍼인가, 아니면 training loop의 실행 계약 자체를 재조립하는 적응 계층인가?
- `accelerator.prepare(...)`는 정확히 무엇을 감추고, 무엇을 여전히 사용자가 알아야 하는가?
- automatic device placement를 켜면 `.to(device)`를 지울 수 있는데, 그렇다고 텐서/optimizer/device 관계를 몰라도 되는가?
- mixed precision을 `Accelerator(mixed_precision=...)`로 쉽게 켤 수 있다는 것과, 실제 수치 안정성/overflow 이해가 필요 없다는 것은 왜 다른가?
- `accelerate launch`는 launcher 복잡도를 줄여 주지만, rank/world size/backend 차이를 완전히 지워 주는가?
- DeepSpeed/FSDP 같은 backend를 Accelerate로 호출할 수 있어도, 왜 다음 단위에서 backend 자체의 메모리/통신 개념을 따로 배워야 하는가?

## 실행 결과 예시
아래 sample output은 `accelerate launch --num_processes 4` 같은 다중 프로세스 실행을 가정한 형태 예시다. 실제 runnable 단계에서는 CPU/GPU 수와 config에 따라 값이 달라질 수 있다.

아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ accelerate launch --num_processes 4 06_training_systems/02_accelerate_workflows/scratch_lab.py
{
  "status": "sample",
  "baseline_loop": {
    "explicit_device_calls": 3,
    "manual_backward": true,
    "launcher": "python train.py"
  },
  "accelerate_loop": {
    "accelerator_device": "cuda:0",
    "distributed_type": "MULTI_GPU",
    "num_processes": 4,
    "device_placement": true,
    "replaced_calls": ["model.to(device)", "inputs.to(device)", "loss.backward()"]
  },
  "observations": [
    "prepare wraps model/optimizer/dataloader in backend-aware containers",
    "same training loop shape survives, but backend details still matter"
  ]
}

$ python 06_training_systems/02_accelerate_workflows/framework_lab.py
{
  "status": "sample",
  "launch_workflow": {
    "config_file_present": true,
    "commands": ["accelerate config", "accelerate test", "accelerate launch train.py"],
    "mixed_precision": "bf16"
  },
  "prepared_objects": {
    "model_wrapper": "Distributed wrapper or backend plugin",
    "optimizer_wrapper": "AcceleratedOptimizer",
    "dataloader_behavior": "sharded_and_device_placed"
  },
  "debug_notes": {
    "optimizer_step_was_skipped": false,
    "sync_gradients": true,
    "manual_rank_logic_removed": "partially"
  },
  "notes": "expected output/sample shape only"
}
```

핵심은 명령 한 줄을 외우는 것이 아니라, **Accelerate가 training loop에서 어떤 보일러플레이트를 흡수하는지**, **그 추상화 아래에 어떤 backend-specific complexity가 여전히 남는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위를 마치면 다음 단위 `06_training_systems/03_deepspeed_zero`를 훨씬 덜 막연하게 볼 수 있다. Accelerate는 launcher와 preparation 단계를 단순화하지만, ZeRO가 실제로 optimizer state / gradient / parameter를 어떻게 shard하는지는 별개의 문제이기 때문이다.

또한 이후 `04_fsdp_checkpointing_and_offload`, `07_data_parallel_grad_accumulation`, `09_profiling_monitoring_and_failure_recovery`를 볼 때도 도움이 된다. 즉, 이 단위는 "분산 학습을 쉽게 켜는 법" 자체보다, **실험 코드와 시스템 backend 사이에 어떤 추상화 계층을 둘 수 있는가** 를 이해하게 만드는 연결 고리다.
