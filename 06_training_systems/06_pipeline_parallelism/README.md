# 06 Pipeline Parallelism

> Status: runnable
>
> 이 단위는 실제 multi-device pipeline runtime을 요구하지 않는다. CPU에서
> deterministic simulation을 실행해 pipeline stage, microbatch schedule,
> bubble/throughput trade-off, activation transfer, partition balance를 관찰한다.

## 왜 이 단위를 배우는가

pipeline parallelism은 "GPU를 더 붙인다"보다 더 구체적으로 **순차 모델의 실행
경로를 pipeline stage로 나눈다**는 뜻이다. data parallel은 같은 모델 복제본이
서로 다른 batch shard를 처리하고, tensor parallel은 레이어 내부 tensor 연산을
나누지만, pipeline parallel은 레이어 묶음을 stage에 배치한 뒤 microbatch를 시간축
위에서 흘려 보낸다.

이 감각은 대형 Transformer 학습에서 중요하다. 한 장치가 모든 레이어와 activation을
들기 어려울 때 stage partition은 memory fit을 도와준다. 하지만 single-batch
latency가 자동으로 줄지는 않는다. pipeline을 채우고 비우는 동안 bubble이 생기고,
stage boundary마다 activation transfer가 필요하며, 가장 느린 stage가 전체
throughput을 묶는다.

## 이번 단위에서 남길 것

- `scratch_lab.py` — 3-stage / 6-microbatch forward fill-drain schedule을 직접
  만들고 metrics JSON + SVG를 남긴다.
- `framework_lab.py` — 4-stage / 8-microbatch 1F1B-style deterministic CPU
  pipeline simulation을 실행한다.
- `analysis.py` — metrics가 없으면 actionable failure를 내고, metrics가 있으면
  stable `analysis.md`와 observed JSON report를 쓴다.
- `THEORY.md`, `PREREQS.md`, `reflection.md` — pipeline stage, microbatch,
  bubble, throughput, activation transfer, partition concern을 한국어 우선으로
  정리한다.
- `lesson.yaml` — runnable 상태와 CPU-safe deterministic 실행 계약을 고정한다.

## 실행 방법

아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python 06_training_systems/06_pipeline_parallelism/scratch_lab.py
python 06_training_systems/06_pipeline_parallelism/framework_lab.py
python 06_training_systems/06_pipeline_parallelism/analysis.py
```

생성되는 주요 산출물:

```text
06_training_systems/06_pipeline_parallelism/artifacts/
├── scratch_metrics.json
├── pipeline_schedule.svg
├── framework_metrics.json
└── analysis_observed.json
```

## 실제 실행 예시

`scratch_lab.py`는 stage 3개와 microbatch 6개를 사용한다. 각 cell은 한 stage가 한
time slot에서 처리하는 microbatch forward 작업이다.

```json
{
  "status": "runnable",
  "simulation": "deterministic_cpu_pipeline_schedule",
  "num_stages": 3,
  "microbatches": 6,
  "schedule_summary": {
    "policy": "forward_pipeline_fill_drain",
    "warmup_slots": 2,
    "steady_state_slots": 4,
    "cooldown_slots": 2,
    "total_time_slots": 8,
    "idle_stage_slots": 6,
    "bubble_fraction": 0.25,
    "throughput_microbatches_per_slot": 0.75
  },
  "activation_transfer": {
    "boundary_count": 2,
    "total_messages": 12,
    "contract": "send forward activations from stage i to i+1 for every microbatch"
  }
}
```

`framework_lab.py`는 실제 PyTorch distributed나 DeepSpeed pipeline engine을 띄우지
않고, dependency-valid 1F1B greedy schedule을 CPU에서 재현한다. forward/backward
operation과 activation 보관량의 차이를 숫자로 읽는 것이 목표다.

```json
{
  "status": "runnable",
  "framework": "deterministic_cpu_pipeline_parallel_sim",
  "schedule_policy": "1F1B_greedy_dependency_sim",
  "num_stages": 4,
  "microbatches": 8,
  "transfers_per_boundary": [
    "forward_activation_send",
    "backward_gradient_recv"
  ],
  "activation_memory_model": {
    "gpipe_peak_saved_microbatches": 8,
    "one_f1b_peak_saved_microbatches": 7
  }
}
```

`analysis.py`는 먼저 두 metrics 파일이 있는지 확인한다. 없으면 다음처럼 바로
고칠 수 있는 실패를 낸다.

```text
Missing required metrics file: artifacts/scratch_metrics.json, artifacts/framework_metrics.json.
Run scratch_lab.py and framework_lab.py first.
```

metrics가 있으면 stable `analysis.md`와 실행별 observed report
`artifacts/analysis_observed.json`를 쓴다.

## 실습 흐름

1. transformer-like layer stack을 놓고 stage boundary를 정한다. 여기서는 의도적으로
   compute가 조금 불균형하게 만들어져 partition balance 질문이 보이게 한다.
2. `scratch_lab.py`에서 forward fill-drain microbatch schedule을 만든다. warmup과
   cooldown의 idle cell을 pipeline bubble로 세고, throughput을
   `microbatches / total_time_slots`로 계산한다.
3. `pipeline_schedule.svg`를 열어 stage별 idle slot과 active slot을 눈으로 확인한다.
4. `activation_transfer.boundary_payload_elements`와 `estimated_bytes`를 보며
   stage boundary가 공짜 선이 아니라 send/recv 계약이라는 점을 확인한다.
5. `framework_lab.py`에서 1F1B-style schedule을 실행하고, GPipe all-forward 방식과
   비교한 activation memory trade-off를 읽는다.
6. 마지막으로 pipeline parallelism이 tensor parallelism, data parallelism,
   FSDP/ZeRO-style sharding과 어떻게 다른 축인지 설명한다.

## 이 단위에서 특히 볼 질문

- pipeline parallelism은 모델 실행 경로를 stage로 나눈다는 점에서 data parallel /
  tensor parallel과 어떻게 다른가?
- stage partition은 왜 메모리 적재 문제를 완화하면서도 single-batch latency를
  자동으로 줄여 주지 않는가?
- microbatch 수가 bubble fraction과 throughput에 어떤 영향을 주며, 왜 무한히
  늘리는 것이 정답이 아닌가?
- GPipe식 스케줄과 1F1B 스케줄은 activation 보관량과 runtime 관찰에서 어떻게
  다르게 보이는가?
- activation transfer와 partition boundary는 왜 단순 레이어 개수 균등 분할보다
  더 중요한 설계 문제가 되는가?

## 다음 단위와의 연결

다음 단위 `06_training_systems/07_data_parallel_grad_accumulation`에서는 같은 모델
복제본을 여러 rank에 두고 effective batch와 optimizer step cadence를 조정하는
data-parallel 축을 본다. 이후 `08_hybrid_parallel_topologies`에서는 pipeline
parallelism을 tensor parallelism, data parallelism, state sharding과 함께 묶어
현실적인 topology로 설계한다.

즉 이 단위의 목표는 "모델을 stage로 나눠 시간축에서 흘린다"는 기준선을 만들고,
나중에 여러 병렬화 축을 섞을 때 pipeline stage split을 독립적으로 설명할 수 있게
하는 것이다.
