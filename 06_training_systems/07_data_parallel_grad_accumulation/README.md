# 07 Data Parallel + Grad Accumulation

> Status: runnable
>
> 이 단위는 실제 multi-GPU DDP를 요구하지 않는다. CPU에서 deterministic simulation을
> 실행해 data parallel replica, grad accumulation window, deferred sync/no_sync,
> all-reduce boundary, optimizer step cadence, loss normalization, gradient clipping,
> memory/throughput trade-off를 안전하게 관찰한다.

## 왜 이 단위를 배우는가

실전 학습에서 가장 자주 마주치는 질문 중 하나는 "OOM 없이 batch를 더 키우려면
어떻게 해야 하는가?"다. 이때 data parallel은 **같은 모델 복제본을 여러 rank에
두고 서로 다른 data shard를 처리하는 batch 축**을 제공하고, grad accumulation은
**optimizer step을 늦춰 더 큰 effective batch를 흉내 내는 스케줄링 축**을 제공한다.

두 기법은 모두 batch budget을 키우는 데 쓰이지만 같은 물건이 아니다.

- data parallel: local batch를 각 rank가 처리하고, microstep마다 global batch는
  `local batch × world size`가 된다.
- grad accumulation: 여러 microstep의 gradient를 buffer에 쌓고, accumulation
  boundary에서 한 번 optimizer step을 한다.
- effective batch: optimizer가 한 번의 update에서 보는 샘플 수이며
  `local batch × world size × grad accumulation steps`로 읽는다.

이 단위의 목표는 숫자를 외우는 것이 아니라 **언제 gradient를 모으고, 언제
all-reduce를 하고, 언제 optimizer/scheduler가 step하는지**를 trace로 설명하는 것이다.

## 이번 단위에서 남길 것

- `scratch_lab.py` — local/global/effective batch 계산, deferred sync/no_sync trace,
  loss normalization, gradient clipping timing, memory model을 직접 계산하고 SVG를 만든다.
- `framework_lab.py` — CPU fallback framework-style data-parallel + accumulation 실행
  계획을 deterministic JSON으로 만든다.
- `analysis.py` — metrics가 없으면 actionable failure를 내고, metrics가 있으면 stable
  `analysis.md`와 observed JSON report를 쓴다.
- `analysis.md` — 실행마다 바뀌지 않는 안정 해석 프레임이다.
- `reflection.md` — 실행 전 예측과 실행 후 해석 질문이다.
- `lesson.yaml` — runnable 상태와 CPU-safe deterministic 실행 계약을 고정한다.

## 실행 방법

아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python 06_training_systems/07_data_parallel_grad_accumulation/scratch_lab.py
python 06_training_systems/07_data_parallel_grad_accumulation/framework_lab.py
python 06_training_systems/07_data_parallel_grad_accumulation/analysis.py
```

생성되는 주요 산출물:

```text
06_training_systems/07_data_parallel_grad_accumulation/artifacts/
├── scratch_metrics.json
├── data_parallel_grad_accumulation.svg
├── framework_metrics.json
└── analysis_observed.json
```

## 실제 실행 예시

`scratch_lab.py`는 local batch 8, world size 4, grad accumulation 4를 사용한다.
따라서 global batch per microstep은 32이고, effective batch per optimizer step은
128이다.

```json
{
  "status": "runnable",
  "cpu_safe_simulation": true,
  "world_size": 4,
  "local_batch_size": 8,
  "grad_accum_steps": 4,
  "global_batch_per_microstep": 32,
  "effective_batch_per_optimizer_step": 128,
  "sync_policy_comparison": {
    "every_step_all_reduce_count": 8,
    "deferred_sync_all_reduce_count": 2,
    "policy": "deferred sync / no_sync until accumulation boundary"
  },
  "loss_normalization": {
    "scale_per_microstep": 0.25
  },
  "gradient_clipping": {
    "recommended_timing": "clip_after_accumulation_boundary"
  }
}
```

`framework_lab.py`는 실제 DDP launcher를 띄우지 않고, framework runtime이 남길 법한
계약을 CPU fallback으로 정리한다.

```json
{
  "status": "runnable",
  "framework": "deterministic_cpu_data_parallel_grad_accum_sim",
  "backend": "cpu_fallback",
  "rank_count": 4,
  "accumulation_steps": 3,
  "global_batch_per_microstep": 24,
  "effective_batch_per_optimizer_step": 72,
  "collectives": [
    "local_backward_no_sync",
    "boundary_all_reduce_gradients",
    "optimizer_step"
  ],
  "scheduler_policy": "scheduler_steps_on_optimizer_step"
}
```

`analysis.py`는 먼저 두 metrics 파일이 있는지 확인한다. 없으면 다음처럼 바로 고칠 수
있는 실패를 낸다.

```text
Missing required metrics file: artifacts/scratch_metrics.json, artifacts/framework_metrics.json.
Run scratch_lab.py and framework_lab.py first.
```

metrics가 있으면 stable `analysis.md`와 실행별 observed report
`artifacts/analysis_observed.json`를 쓴다.

## 실습 흐름

1. single-rank 학습 루프를 기준으로 forward/backward/optimizer step 순서를 떠올린다.
2. data parallel 상황을 가정해 rank 4개가 각각 local batch 8을 처리하면 global batch가
   32가 된다는 것을 계산한다.
3. grad accumulation steps 4를 넣어 optimizer step이 microstep 4개마다 한 번만
   발생하는 trace를 확인한다.
4. accumulation window 안에서는 `no_sync`로 local backward만 수행하고, boundary에서
   deferred sync/all-reduce와 optimizer step이 함께 일어나는 cadence를 읽는다.
5. loss normalization은 microstep loss를 1/4로 scale하고, gradient clipping은
   accumulation boundary 뒤 aggregate gradient에 적용해야 함을 확인한다.
6. 큰 local batch와 작은 local batch + grad accumulation의 memory peak가 왜 다른지
   `memory_model_mb`로 비교한다.
7. effective batch가 커져도 forward/backward 횟수가 늘어 throughput이 자동으로
   좋아지지는 않는다는 점을 framework simulation의 throughput model과 연결한다.

## 이 단위에서 특히 볼 질문

- data parallel은 batch를 어떤 축으로 키우고, grad accumulation은 optimizer step을
  어떤 방식으로 늦추는가?
- local batch, global batch, effective batch를 같은 말처럼 쓰면 어떤 해석 오류가 생기는가?
- every-step all-reduce와 deferred sync/no_sync는 communication cadence에서 어떻게
  다르게 보이는가?
- loss normalization, gradient clipping, scheduler step timing은 왜 accumulation
  boundary와 함께 해석해야 하는가?
- 큰 local batch와 작은 local batch + accumulation은 memory, utilization,
  optimizer noise 측면에서 어떻게 다른가?
- later hybrid parallel setup에서 data-parallel 축은 왜 여전히 batch budget과
  optimizer step cadence의 기준선으로 남는가?

## CPU/GPU 안전성

이 단위의 canonical path는 CPU-safe deterministic simulation이다. GPU가 있어도 필수
실행은 CPU에서 동일하게 돌아가며, 실제 torchrun/DDP smoke는 후속 운영 실습에서만
별도로 다룬다. 따라서 CI와 로컬 학습 환경 모두에서 같은 JSON과 SVG를 기대할 수 있다.

## 다음 단위와의 연결

다음 단위 `06_training_systems/08_hybrid_parallel_topologies`에서는 data parallel,
tensor parallel, pipeline parallel, sharding을 왜 한 번에 섞어야 하는지 본격적으로
다룬다. 이 단위에서 local/global/effective batch와 optimizer step cadence를 분명히
잡아 두면, later hybrid setup에서도 **어느 축이 batch budget을 책임지고 어느 축이
model fit을 책임지는지**를 훨씬 또렷하게 분리할 수 있다.
