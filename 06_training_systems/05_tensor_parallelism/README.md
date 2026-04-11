# 05 Tensor Parallelism

> Status: runnable
>
> 이 단위는 실제 GPU tensor-parallel runtime을 요구하지 않는다. 대신 CPU에서 순수
> Python으로 deterministic하게 column-parallel / row-parallel linear split을
> 재현해, 각 rank가 어떤 **행렬 shard**와 **activation shard**를 들고 어떤
> communication overhead를 치르는지 관찰한다.

## 왜 이 단위를 배우는가

텐서 병렬(tensor parallelism)은 모델 병렬(model parallelism)의 한 형태지만,
핵심은 "모델 파일을 여러 곳에 둔다"가 아니라 **레이어 내부 계산을 여러 rank가
나눠 맡는다**는 점이다.

DDP, ZeRO, FSDP는 주로 **모델 상태를 어떻게 복제하거나 shard할 것인가**를
다룬다. 하지만 초대형 Transformer로 가면 어떤 시점에는 "상태를 어디에 둘
것인가"만으로는 부족하고, **레이어 내부 행렬 연산 자체를 여러 장치가 함께
계산해야 하는 순간**이 온다. Tensor parallelism은 바로 그 지점을 다루며, 하나의
linear/attention/feed-forward 블록을 여러 rank에 쪼개서 계산하는 감각을 만든다.

이 단위를 이해하면 "모델이 너무 커서 한 장치에 안 들어간다"를 단순 OOM 문제로만
보지 않고, **어떤 차원을 나누면 계산과 메모리를 함께 분산할 수 있는가**라는
시스템 질문으로 다시 보게 된다. 또한 FSDP, pipeline parallelism, hybrid
parallelism을 비교할 때도 "상태를 나누는가, 레이어를 나누는가, 레이어 안을
나누는가"를 구분하는 기준선이 된다.

## 이번 단위에서 남길 것

- `scratch_lab.py` — 순수 Python matmul로 column-parallel / row-parallel split을
  직접 계산하고 metrics JSON + SVG를 남긴다.
- `framework_lab.py` — Megatron-style tensor parallel block을 흉내 내는
  deterministic CPU simulation을 실행한다.
- `analysis.py` — metrics가 없으면 actionable failure를 내고, metrics가 있으면
  stable `analysis.md`와 observed JSON report를 갱신한다.
- `THEORY.md`, `PREREQS.md`, `reflection.md` — tensor/model parallel split,
  matrix shard, activation shard, communication overhead, FSDP/pipeline과의 관계를
  한국어 우선으로 정리한다.
- `lesson.yaml` — runnable 상태와 CPU-safe deterministic 실행 계약을 고정한다.

## 실행 방법

아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python 06_training_systems/05_tensor_parallelism/scratch_lab.py
python 06_training_systems/05_tensor_parallelism/framework_lab.py
python 06_training_systems/05_tensor_parallelism/analysis.py
```

생성되는 주요 산출물:

```text
06_training_systems/05_tensor_parallelism/artifacts/
├── scratch_metrics.json
├── tensor_parallelism_shards.svg
├── framework_metrics.json
└── analysis_observed.json
```

## 실제 실행 예시

`scratch_lab.py`는 작은 입력 `[3, 8]`과 world size 4를 사용해 dense matmul과
sharded matmul의 결과가 같다는 것을 확인한다.

```json
{
  "status": "runnable",
  "tp_world_size": 4,
  "input_shape": [3, 8],
  "column_parallel": {
    "global_weight_shape": [8, 16],
    "per_rank_weight_shape": [8, 4],
    "per_rank_activation_shape": [3, 4],
    "collective": "all_gather_if_full_activation_required"
  },
  "row_parallel": {
    "global_weight_shape": [16, 6],
    "per_rank_weight_shape": [4, 6],
    "per_rank_activation_shape": [3, 4],
    "collective": "all_reduce_sum"
  },
  "max_abs_diff_vs_dense": 0
}
```

`framework_lab.py`는 실제 Megatron/DeepSpeed runtime을 띄우지 않고, 같은 개념을
framework-style 실행 계획으로 읽게 만든다.

```json
{
  "status": "runnable",
  "framework": "deterministic_cpu_tensor_parallel_sim",
  "tp_world_size": 4,
  "attention_partition": {
    "num_heads_total": 8,
    "heads_per_rank": 2,
    "hidden_size": 8
  },
  "collectives_per_block": [
    "all_gather_activations",
    "all_reduce_partial_outputs"
  ]
}
```

`analysis.py`는 먼저 두 metrics 파일이 있는지 확인한다. 없으면 다음처럼 바로
고칠 수 있는 실패를 낸다.

```text
Missing required metrics file: artifacts/scratch_metrics.json, artifacts/framework_metrics.json.
Run scratch_lab.py and framework_lab.py first.
```

metrics가 있으면 stable `analysis.md`와 실행별 observed report를 쓴다.

## 실습 흐름

1. single-device dense linear를 기준으로 잡고, 큰 행렬곱이 hidden dimension /
   intermediate dimension에서 왜 병목이 되는지 확인한다.
2. column-parallel linear에서 output feature 차원을 4개 rank로 나눠, 각 rank의
   weight shard `[8, 4]`와 activation shard `[3, 4]`를 관찰한다.
3. row-parallel linear에서 input feature 차원을 4개 rank로 나눠, 각 rank의 partial
   output을 `all_reduce_sum`으로 합쳐 dense 결과와 일치시키는 과정을 본다.
4. attention head partition에서는 전체 8개 head를 rank당 2개씩 맡기는 식의
   model parallel split을 읽는다.
5. metrics의 `communication_overhead.estimated_bytes`와
   `throughput_model.communication_share`를 보며, 메모리 절감이 왜 공짜가 아닌지
   정리한다.
6. 마지막으로 FSDP/ZeRO는 상태 sharding, pipeline parallelism은 stage split,
   tensor parallelism은 intra-layer split이라는 차이를 말로 설명한다.

## 이 단위에서 특히 볼 질문

- tensor parallelism은 DDP/FSDP/ZeRO처럼 상태를 나누는 접근과 무엇이 근본적으로
  다른가?
- column-parallel과 row-parallel은 각각 어떤 텐서 차원을 나누며, 어떤 시점에
  collective communication이 필요한가?
- activation shard를 계속 쥔 채 다음 연산으로 넘길 수 있는 경우와, full
  activation을 다시 모아야 하는 경우는 어떻게 다른가?
- tensor parallelism은 메모리를 줄여 주는데도 왜 interconnect bandwidth와 latency에
  민감한가?
- pipeline parallelism, sharding, hybrid parallelism을 볼 때 tensor parallel은 어떤
  역할의 퍼즐 조각인가?

## 다음 단위와의 연결

다음 단위 `06_training_systems/06_pipeline_parallelism`에서는 레이어 내부를 나누는
대신 **레이어 묶음을 stage로 쪼개는 방법**을 본다. 즉 tensor parallelism이 "한
레이어를 여러 rank가 함께 계산하는 법"이라면, pipeline parallelism은 "서로 다른
레이어 구간을 여러 rank가 순차로 맡는 법"을 다룬다.

그 다음 `06_training_systems/08_hybrid_parallel_topologies`에서는 data parallel +
tensor parallel + pipeline parallel + FSDP/ZeRO-style sharding을 한 번에 묶는
현실적인 조합으로 넘어간다. 따라서 이 단위는 이후 분산 전략들을 비교할 때,
**intra-layer split이라는 독립된 축**을 머릿속에 고정해 주는 연결 고리다.
