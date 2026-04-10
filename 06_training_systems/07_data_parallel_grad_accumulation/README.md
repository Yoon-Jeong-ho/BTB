# 07 Data Parallel + Grad Accumulation

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 runnable/applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
실전 학습에서 가장 자주 마주치는 질문 중 하나는 "OOM 없이 batch를 더 키우려면 어떻게 해야 하는가?"다. 이때 data parallel은 **같은 모델 복제본을 여러 rank에 두고 서로 다른 데이터 shard를 처리하는 축**을 제공하고, grad accumulation은 **optimizer step을 늦춰 더 큰 effective batch를 흉내 내는 운영 축**을 제공한다. 이 단위는 이 둘을 한 번에 보면서, "GPU를 더 쓰는 것"과 "step 타이밍을 늦추는 것"이 어떻게 다른지 구분하게 만든다.

또한 이후 대형 모델 학습에서는 batch budget, communication cadence, memory ceiling을 함께 조절해야 한다. 이 단위를 이해하면 local batch / global batch / effective batch를 섞어 말하는 혼란이 줄고, 왜 어떤 실험은 GPU를 늘려도 throughput이 기대만큼 안 오르고 어떤 실험은 accumulation만 늘려도 optimizer dynamics가 달라 보이는지 더 정확히 읽게 된다. 즉 이 단위는 분산 학습을 단순 launch 문제가 아니라 **step scheduling과 communication policy까지 포함한 시스템 문제**로 보는 관문이다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- data parallel intuition, effective batch math, sync cadence 감각을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - per-rank local batch / global batch / effective batch 계산 요약
  - accumulation step별 forward/backward/optimizer step cadence trace
  - sync-every-step vs deferred-sync 관찰 메모
  - throughput / memory / optimizer-noise trade-off 비교 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. single-GPU 기준으로 local batch를 키우다가 메모리 한계에 걸리는 상황을 먼저 떠올리며, 왜 batch size를 늘리는 문제가 곧바로 optimizer step / noise scale / throughput 문제로 이어지는지 정리한다.
2. 같은 모델을 여러 rank에 복제한 data parallel 상황을 가정하고, 각 rank가 다른 mini-batch shard를 처리한 뒤 gradient를 맞추면 global batch가 어떻게 커지는지 본다.
3. local batch는 그대로 둔 채 optimizer step을 여러 backward 뒤로 미루는 grad accumulation을 도입해, 왜 effective batch가 `local_batch × world_size × accumulation_steps`로 읽히는지 계산한다.
4. accumulation 중간 step에서는 gradient를 계속 더하고 optimizer step은 마지막에만 하는 흐름을 따라가며, loss normalization과 gradient scale 해석이 왜 중요해지는지 관찰한다.
5. every-step synchronization과 accumulation window 안에서의 deferred synchronization(예: `no_sync`)을 비교하며, communication 횟수 감소가 언제 도움이 되고 언제 stale gradient / 긴 step latency처럼 느껴지는지 본다.
6. 마지막에는 "큰 local batch"와 "작은 local batch + accumulation"이 메모리, throughput, optimizer dynamics에서 어떤 차이를 만드는지 비교하고, 다음 단위 `06_training_systems/08_hybrid_parallel_topologies`와 연결한다.

## 이 단위에서 특히 볼 질문
- data parallel은 batch를 어떤 축으로 키우고, grad accumulation은 optimizer step을 어떤 방식으로 늦추는가?
- local batch, global batch, effective batch를 서로 같은 말처럼 쓰면 어떤 해석 오류가 생기는가?
- accumulation step을 늘리면 OOM을 피할 수 있는데도 왜 wall-clock throughput이 자동으로 좋아지지는 않는가?
- DDP에서 every-step all-reduce와 accumulation window 뒤의 sync는 communication cadence에서 어떻게 다르게 보이는가?
- 큰 local batch와 작은 local batch + accumulation은 메모리 사용량, kernel efficiency, optimizer noise 측면에서 어떻게 다른가?
- later hybrid parallel setup에서 data-parallel 축은 왜 여전히 batch budget과 optimizer-step cadence의 기준선으로 남는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ torchrun --standalone --nproc_per_node=4 06_training_systems/07_data_parallel_grad_accumulation/scratch_lab.py
{
  "status": "sample",
  "local_batch_size": 8,
  "world_size": 4,
  "grad_accum_steps": 4,
  "global_batch_per_microstep": 32,
  "effective_batch_per_optimizer_step": 128,
  "optimizer_step_every": 4,
  "per_rank_trace": [
    {"rank": 0, "microstep": 1, "sync": false, "optimizer_step": false},
    {"rank": 0, "microstep": 2, "sync": false, "optimizer_step": false},
    {"rank": 0, "microstep": 3, "sync": false, "optimizer_step": false},
    {"rank": 0, "microstep": 4, "sync": true, "optimizer_step": true}
  ],
  "notes": "expected output/sample shape only"
}

$ torchrun --standalone --nproc_per_node=4 06_training_systems/07_data_parallel_grad_accumulation/framework_lab.py
{
  "status": "sample",
  "framework": "ddp + grad accumulation",
  "memory_profile": {
    "local_batch_limit": "fits_at_8",
    "same_effective_batch_with_accumulation": "8 x 4 ranks x 4 accum = 128"
  },
  "runtime_observations": {
    "sync_policy": "defer_allreduce_until_accum_boundary",
    "step_time": "longer_optimizer_interval",
    "throughput_risk": "more_forward_backward_per_optimizer_step",
    "optimizer_noise": "closer_to_large_batch_regime"
  },
  "notes": "expected output/sample shape only"
}
```

핵심은 숫자를 외우는 것이 아니라, **언제 gradient를 모으고 언제 optimizer step을 하는지**, **memory ceiling 때문에 local batch를 못 키울 때 accumulation이 어떤 우회로를 주는지**, **effective batch 증가가 통신·throughput·optimization에 어떤 흔적을 남기는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/08_hybrid_parallel_topologies`에서는 data parallel, tensor parallel, pipeline parallel, sharding을 왜 한 번에 섞어야 하는지 본격적으로 다룬다. 이 단위에서 local/global/effective batch와 step cadence를 분명히 잡아 두면, later hybrid setup에서도 **어느 축이 batch budget을 책임지고 어느 축이 model fit을 책임지는지**를 훨씬 또렷하게 분리할 수 있다.

또한 `06_training_systems/09_profiling_monitoring_and_failure_recovery`로 가면 throughput 저하, sync stall, accumulation misconfiguration 같은 현상을 운영 로그와 profiler 흔적으로 읽게 된다. 따라서 이 단위는 단순 batch math 정리를 넘어서, 이후 training systems 전체에서 **communication cadence와 optimizer-step cadence를 해석하는 기준선** 역할을 한다.
