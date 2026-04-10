# 03 DeepSpeed ZeRO

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
단일 GPU나 단순 DDP만으로는 모델 크기와 batch를 조금만 키워도 **optimizer state / gradient / parameter 메모리**가 빠르게 한계에 닿는다. DeepSpeed ZeRO는 이 병목을 "더 큰 GPU를 사자"로만 해결하지 않고, **어떤 상태를 어느 rank에 나눠 들고 있을지** 다시 설계하는 방식으로 풀어낸다. 이 단위는 distributed training을 throughput 문제만이 아니라 **메모리 배치 문제**로 보게 만들고, 이후 FSDP·offload·hybrid parallelism을 읽을 때 기준점이 되는 첫 분산 메모리 최적화 단위다.

또한 later LLM 학습에서는 effective batch, optimizer choice, offload 여부, communication 비용이 모두 한꺼번에 얽힌다. ZeRO를 먼저 이해하면 "왜 stage가 올라갈수록 메모리는 줄지만 orchestration은 복잡해지는가"를 납득할 수 있고, 단순 gradient accumulation만으로는 안 풀리는 대형 모델 운영 감각도 함께 잡을 수 있다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- ZeRO partitioning intuition과 메모리 분해 관점을 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - stage별 memory breakdown 비교 표
  - optimizer/gradient/parameter shard 관찰 메모
  - communication vs memory trade-off 요약
  - DeepSpeed config 핵심 필드 점검표

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 "왜 DDP는 같은 모델을 rank마다 통째로 들고 있어 메모리가 빨리 터지는가"를 작은 수치 예시로 다시 적는다.
2. ZeRO Stage 1, 2, 3를 순서대로 비교하며 optimizer state, gradient, parameter가 각각 언제 partition되는지 본다.
3. 동일한 모델/optimizer 조건에서 stage가 올라갈수록 per-rank memory가 어떻게 줄고, 대신 어떤 collective communication이 늘어나는지 관찰한다.
4. DeepSpeed config에서 `zero_optimization`, gradient accumulation, mixed precision, offload 관련 필드가 어떤 운영 의미를 갖는지 읽는다.
5. 단순 gradient accumulation 또는 작은 batch와 비교해, ZeRO가 해결하는 문제와 해결하지 못하는 문제를 분리해 본다.
6. 마지막에는 ZeRO intuition을 바탕으로 다음 단위 `06_training_systems/04_fsdp_checkpointing_and_offload`에서 FSDP와 offload를 왜 따로 봐야 하는지 연결한다.

## 이 단위에서 특히 볼 질문
- DDP에서는 무엇이 모든 rank에 중복되고, ZeRO는 그중 어느 상태부터 나눠 들기 시작하는가?
- optimizer state, gradient, parameter는 왜 메모리 성격과 통신 타이밍이 서로 다른가?
- Stage 1/2/3가 단순히 "숫자가 클수록 좋다"가 아닌 이유는 무엇인가?
- gradient accumulation이나 작은 micro-batch로 줄일 수 있는 병목과, ZeRO가 직접 해결하는 병목은 어떻게 다른가?
- DeepSpeed가 큰 모델 학습에서 중요한 이유는 단순 메모리 절약 외에 무엇인가?
- 이후 FSDP, checkpointing, offload를 볼 때 ZeRO intuition이 어떤 기준선 역할을 하는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 06_training_systems/03_deepspeed_zero/scratch_lab.py
{
  "status": "sample",
  "model": {
    "param_count_millions": 1300,
    "dtype": "bf16",
    "world_size": 8
  },
  "memory_breakdown_gb": {
    "ddp_full_replica": {
      "parameters": 2.6,
      "gradients": 2.6,
      "optimizer_states": 10.4
    },
    "zero_stage_1_per_rank": {
      "parameters": 2.6,
      "gradients": 2.6,
      "optimizer_states": 1.3
    },
    "zero_stage_2_per_rank": {
      "parameters": 2.6,
      "gradients": 0.325,
      "optimizer_states": 1.3
    },
    "zero_stage_3_per_rank": {
      "parameters": 0.325,
      "gradients": 0.325,
      "optimizer_states": 1.3
    }
  },
  "notes": [
    "sample numbers for intuition only",
    "communication cost rises as more states are partitioned"
  ]
}

$ python 06_training_systems/03_deepspeed_zero/framework_lab.py
{
  "status": "sample",
  "deepspeed_config": {
    "zero_stage": 2,
    "gradient_accumulation_steps": 8,
    "bf16": true,
    "offload_optimizer": false
  },
  "observations": {
    "optimizer_state_partitioned": true,
    "gradient_partitioned": true,
    "parameter_partitioned": false,
    "dominant_collectives": ["reduce_scatter", "all_gather"]
  },
  "expected_logs": {
    "global_batch": 256,
    "micro_batch_per_gpu": 4,
    "memory_saved_vs_ddp": "moderate",
    "throughput_tradeoff": "communication_overhead_visible"
  }
}
```

핵심은 숫자 하나를 외우는 것이 아니라, **각 state가 어디에 저장되는지**, **stage 변화가 메모리와 통신에 어떤 다른 흔적을 남기는지**, **DeepSpeed config가 운영 의미로 어떻게 읽히는지**를 해석하는 것이다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/04_fsdp_checkpointing_and_offload`에서는 ZeRO의 메모리 분산 감각을 한 단계 더 밀어, **parameter sharding을 언제/어떻게 runtime에 풀어 쓰는가**, **checkpointing과 offload를 함께 넣으면 어떤 운영 trade-off가 생기는가**를 본다. 즉 이 단위가 "어떤 상태를 나눌 것인가"를 이해하게 만든다면, 다음 단위는 그 위에서 **parameter lifecycle과 저장/복구 전략까지 포함한 더 적극적인 메모리 운영**으로 넘어가는 연결 고리다.
