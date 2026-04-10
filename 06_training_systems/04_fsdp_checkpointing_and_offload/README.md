# 04 FSDP, Checkpointing, and Offload

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`06_training_systems/03_deepspeed_zero`에서 이미 보았듯이, 대형 모델 학습의 병목은 단순히 연산량이 아니라 **어떤 상태를 어느 시점에 어느 장치에 두는가**라는 메모리 운영 문제다. FSDP(Fully Sharded Data Parallel)는 이 질문을 더 강하게 밀어붙여, parameter/gradient/optimizer state를 rank 사이에 나눠 들고 필요할 때만 모아 쓰는 runtime 전략으로 이해하는 것이 핵심이다. 이 단위는 DDP와 ZeRO를 넘어, **모델 복제 대신 shard lifecycle 자체를 읽는 감각**을 만든다.

또한 실제 대형 학습에서는 sharding만으로 끝나지 않는다. activation checkpointing은 메모리를 아끼는 대신 재계산을 늘리고, CPU/NVMe offload는 GPU 메모리를 비우는 대신 전송 지연을 끌어온다. checkpoint save/load도 full state로 저장할지, sharded state로 저장할지에 따라 복구 흐름과 portability가 달라진다. 이 단위는 그래서 FSDP를 API 이름으로 외우는 것이 아니라, **memory-compute-I/O trade-off를 동시에 읽는 시스템 단위**로 다룬다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- FSDP sharding intuition, activation checkpointing, offload trade-off를 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - FSDP shard / all-gather / reduce-scatter 관찰 메모
  - activation checkpointing on/off memory-vs-step-time 비교 표
  - CPU offload / no-offload 비교 메모
  - full state dict vs sharded state dict save/load 체크리스트

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 DDP와 ZeRO Stage 3 직관을 다시 꺼내 보며, FSDP가 "각 rank가 모델 전체를 항상 들고 있지 않아도 되는가"라는 질문에 어떻게 답하는지 작은 메모리 그림으로 정리한다.
2. FSDP에서 parameter shard가 forward 직전에 all-gather되고, backward 이후 다시 shard/정리되는 lifecycle을 따라가며 **언제 full parameter view가 잠깐 나타나고 언제 사라지는지** 본다.
3. activation checkpointing을 켰을 때 저장되는 activation이 줄고 대신 recomputation이 늘어난다는 점을 step timeline 위에서 비교한다.
4. CPU offload 또는 더 느린 저장장치 offload를 가정해, GPU 메모리는 줄지만 host-device transfer와 step latency가 어떻게 커질 수 있는지 정리한다.
5. checkpoint 저장 시 full state dict와 sharded state dict를 각각 어떤 상황에서 쓰는지, resume/portability/debugging 관점에서 비교한다.
6. 마지막에는 이 단위가 이후 `06_training_systems/05_tensor_parallelism`, `08_hybrid_parallel_topologies`에서 왜 필요한지 연결하며, "메모리 shard 전략"과 "연산 자체를 쪼개는 전략"을 분리해 본다.

## 이 단위에서 특히 볼 질문
- FSDP는 DDP/ZeRO와 비교해 parameter를 언제 모으고 언제 다시 쪼개는 runtime으로 이해하면 좋은가?
- activation checkpointing은 어떤 메모리를 줄이고, 그 대가로 어떤 재계산 비용을 만든다고 봐야 하는가?
- CPU offload는 "메모리 해결책"인 동시에 왜 step time과 안정성 측면의 새 병목이 되는가?
- full state dict와 sharded state dict는 저장/복구/이식성 관점에서 어떤 서로 다른 운영 계약을 가지는가?
- `state_dict_type`, auto wrap policy, mixed precision, offload 설정은 왜 따로가 아니라 한 시스템 그림 안에서 읽어야 하는가?
- FSDP를 이해한 뒤 tensor parallelism을 보면, "상태를 쪼개는 것"과 "연산을 쪼개는 것"의 차이가 어떻게 더 또렷해지는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ torchrun --standalone --nproc_per_node=4 06_training_systems/04_fsdp_checkpointing_and_offload/scratch_lab.py
{
  "status": "sample",
  "fsdp": {
    "world_size": 4,
    "sharding_strategy": "FULL_SHARD",
    "mixed_precision": "bf16",
    "activation_checkpointing": true,
    "cpu_offload": false
  },
  "memory_observation": {
    "before_wrap_gb": 21.4,
    "after_wrap_per_rank_gb": 6.8,
    "peak_forward_gather_gb": 9.7,
    "peak_backward_gb": 10.4
  },
  "notes": [
    "sample numbers for intuition only",
    "checkpointing reduces activation memory but increases recompute"
  ]
}

$ torchrun --standalone --nproc_per_node=4 06_training_systems/04_fsdp_checkpointing_and_offload/framework_lab.py
{
  "status": "sample",
  "checkpoint_plan": {
    "state_dict_type": "sharded",
    "save_rank0_only": false,
    "resume_target": "same_world_size_or_reshard_capable_runtime"
  },
  "offload_comparison": {
    "cpu_offload": {
      "peak_gpu_memory_gb": 5.9,
      "step_time_ms": 1820
    },
    "no_offload": {
      "peak_gpu_memory_gb": 8.4,
      "step_time_ms": 1210
    }
  },
  "expected_logs": {
    "all_gather_visible": true,
    "reduce_scatter_visible": true,
    "full_state_export_needed_for_portable_inference": true
  }
}
```

핵심은 숫자를 외우는 것이 아니라, **FSDP shard lifecycle이 메모리를 어떻게 바꾸는지**, **checkpointing과 offload가 어떤 다른 비용을 가져오는지**, **state dict 저장 방식이 복구 전략에 어떤 흔적을 남기는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/05_tensor_parallelism`에서는 메모리를 아끼기 위해 상태를 shard하는 대신, **레이어 내부 연산 자체를 여러 장치로 나누는 방식**을 본다. 그래서 이 단위는 "상태 배치와 복구 전략"의 관점, 다음 단위는 "연산 분할과 통신 경로"의 관점이라고 구분해 두면 좋다.

이후 `06_training_systems/08_hybrid_parallel_topologies`로 가면 FSDP/ZeRO, tensor parallel, pipeline parallel이 한꺼번에 섞인다. 따라서 이 단위에서 FSDP checkpoint/offload 감각을 잡아 두면, 나중에 hybrid setup을 볼 때도 **어떤 병목이 메모리 때문이고 어떤 병목이 연산 분할 때문인지** 더 선명하게 설명할 수 있다.
