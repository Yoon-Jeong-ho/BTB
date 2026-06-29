# 04 FSDP, Checkpointing, and Offload

> Status: runnable
>
> 이 단위는 실제 FSDP/GPU 없이 실행되는 CPU-safe deterministic simulation이다. 목적은 PyTorch FSDP API를 그대로 실행하는 것이 아니라, FSDP sharding, activation checkpointing, CPU offload, full/sharded checkpoint loading의 **숫자 계약과 운영 판단**을 작게 관찰하는 것이다.

## 왜 이 단위를 배우는가
대형 모델 학습에서 OOM은 단순히 “GPU가 작다”가 아니라 **parameter, gradient, optimizer state, activation이 어느 순간 어느 장치에 resident한가**의 문제다. FSDP(Fully Sharded Data Parallel)는 DDP처럼 모델 전체를 각 rank에 복제해 두는 그림을 깨고, 상태를 shard로 들다가 필요한 순간에만 full parameter view를 all-gather한다.

이 단위는 세 가지 교환 관계를 함께 본다.

1. **FSDP sharding**: per-rank memory를 줄이지만 all-gather / reduce-scatter 통신과 peak 순간을 만든다.
2. **Activation checkpointing**: activation 저장량을 줄이지만 backward 때 recomputation을 늘린다.
3. **CPU offload**: GPU peak를 낮추지만 host-device transfer latency를 step time에 남긴다.

마지막으로 checkpoint 저장/복구에서는 **full state dict**와 **sharded state dict**의 차이를 비교한다. full state는 export/debug가 쉽지만 load peak가 크고, sharded state는 resume memory가 낮지만 world size 변경 또는 다른 runtime 이식 시 merge/reshard 계약이 필요하다.

## 파일 구성
- `scratch_lab.py` — FSDP shard lifecycle, activation checkpointing, CPU offload memory trade-off를 직접 계산하고 SVG를 만든다.
- `framework_lab.py` — 실제 FSDP 대신 deterministic framework-style checkpoint/offload policy metrics를 만든다.
- `analysis.py` — 두 metrics 파일이 없으면 실행 방법을 알려 주며 실패하고, 있으면 관측 보고서를 `artifacts/analysis-manual/latest_report.md`에 쓴다.
- `analysis.md` — 실행해도 바뀌지 않는 안정 분석 가이드다.
- `reflection.md` — 실행 전 예측과 실행 후 해석 질문이다.
- `lesson.yaml` — runnable 상태와 required outputs를 고정한다.

## 실행 방법
clone한 저장소 루트에서 실행한다.

```bash
python3 06_training_systems/04_fsdp_checkpointing_and_offload/scratch_lab.py
python3 06_training_systems/04_fsdp_checkpointing_and_offload/framework_lab.py
python3 06_training_systems/04_fsdp_checkpointing_and_offload/analysis.py
```

생성 산출물:

```text
06_training_systems/04_fsdp_checkpointing_and_offload/artifacts/
├── scratch-manual/
│   ├── fsdp_memory_tradeoffs.svg
│   └── metrics.json
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시
`scratch_lab.py`는 순수 산술 simulation을 출력한다. 핵심 숫자는 DDP full replica 기준선에서 FSDP/checkpoint/offload를 단계적으로 켰을 때 peak가 어떻게 내려가는지다.

```text
$ python3 06_training_systems/04_fsdp_checkpointing_and_offload/scratch_lab.py
{
  "cpu_safe_simulation": true,
  "world_size": 4,
  "sharding_strategy": "FULL_SHARD",
  "ddp_full_replica_per_rank_mb": 544.0,
  "fsdp_forward_peak_gpu_mb": 328.0,
  "fsdp_checkpointed_peak_gpu_mb": 232.0,
  "cpu_offload_gpu_peak_mb": 196.0,
  "activation_checkpoint_saving_ratio": 0.6,
  "checkpoint_recompute_multiplier": 1.28,
  "lifecycle_events": [
    "rank_holds_parameter_shard",
    "all_gather_full_params",
    "reduce_scatter_gradients",
    "optimizer_step_on_shard"
  ],
  "figure_path": "artifacts/scratch-manual/fsdp_memory_tradeoffs.svg"
}
```

`framework_lab.py`는 checkpoint/offload 정책을 FSDP-style runtime 결과처럼 정리한다.

```text
$ python3 06_training_systems/04_fsdp_checkpointing_and_offload/framework_lab.py
{
  "backend": "cpu-simulated-fsdp-checkpoint-offload",
  "rank_count": 4,
  "best_resume_mode_by_peak": "sharded_state_dict",
  "portable_export_mode": "full_state_dict",
  "state_dict_modes": {
    "full_state_dict": {
      "file_count": 1,
      "portable_export": true,
      "load_peak_mb": 384.0
    },
    "sharded_state_dict": {
      "file_count": 4,
      "portable_export": false,
      "load_peak_mb": 120.0
    }
  },
  "offload_policy": {
    "none": {"peak_gpu_memory_mb": 232.0, "step_time_ms": 1280},
    "cpu_optimizer_offload": {"peak_gpu_memory_mb": 196.0, "step_time_ms": 1510}
  }
}
```

`analysis.py`는 두 metrics를 읽어 관측 보고서를 만든다.

```text
$ python3 06_training_systems/04_fsdp_checkpointing_and_offload/analysis.py
# 04 FSDP Checkpointing and Offload 실행 관측

## 관측 결과
- DDP full replica per-rank memory: `544.0 MB`
- FSDP peak with activation checkpointing: `232.0 MB`
- CPU optimizer offload peak: `196.0 MB`
- full state dict load peak: `384.0 MB`
- sharded state dict load peak: `120.0 MB`
```

metrics 없이 `analysis.py`를 먼저 실행하면 다음처럼 실패한다.

```text
필수 metrics 파일이 없습니다: artifacts/scratch-manual/metrics.json, artifacts/framework-manual/metrics.json. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.
```

## 관찰 포인트
- FSDP의 steady-state memory와 forward peak memory를 분리해서 본다.
- activation checkpointing은 activation memory를 줄이지만 step time에는 recompute multiplier를 남긴다.
- CPU offload는 GPU peak를 더 줄이지만 transfer latency를 추가한다.
- full state dict는 export/debug에 유리하고, sharded state dict는 동일 계열 학습 resume에 유리하다.
- 이 simulation은 실제 bandwidth, NCCL collective, CUDA allocator를 재현하지 않는다. 대신 실제 프로파일링 전에 어떤 metric을 봐야 하는지 안전하게 연습한다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/05_tensor_parallelism`에서는 상태를 shard하는 FSDP와 달리, layer 내부 연산 자체를 여러 장치로 나누는 방법을 본다. 이 단위에서 “상태 배치와 checkpoint 복구 계약”을 잡아 두면, hybrid parallelism에서 어떤 병목이 memory residency 문제이고 어떤 병목이 compute partition 문제인지 더 선명하게 구분할 수 있다.
