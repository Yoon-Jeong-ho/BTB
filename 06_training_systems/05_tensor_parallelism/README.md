# 05 Tensor Parallelism

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
DDP, ZeRO, FSDP는 주로 **모델 상태를 어떻게 복제하거나 shard할 것인가**를 다룬다. 하지만 초대형 Transformer로 가면 어떤 시점에는 "상태를 어디에 둘 것인가"만으로는 부족하고, **레이어 내부 행렬 연산 자체를 여러 장치가 함께 계산해야 하는 순간**이 온다. Tensor parallelism은 바로 그 지점을 다루며, 하나의 linear/attention/feed-forward 블록을 여러 GPU에 쪼개서 계산하는 감각을 만든다.

이 단위를 이해하면 "모델이 너무 커서 한 장치에 안 들어간다"를 단순 OOM 문제로만 보지 않고, **어떤 차원을 나누면 계산과 메모리를 함께 분산할 수 있는가**라는 시스템 질문으로 다시 보게 된다. 또한 이후 pipeline parallelism, hybrid parallel topology를 볼 때도 "레이어를 나누는가, 레이어 안을 나누는가, 상태만 나누는가"를 구분하는 기준선이 된다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- tensor/model parallel intuition과 intra-layer split 감각을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - per-rank weight shard / activation shard shape 요약
  - column-parallel / row-parallel linear 관찰 메모
  - attention head 분할과 collective communication 로그 요약
  - latency vs memory trade-off 관찰 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. single-GPU linear layer와 Transformer block을 먼저 떠올리며, 큰 행렬곱이 왜 hidden dimension / intermediate dimension에서 메모리와 runtime 병목을 만들기 쉬운지 정리한다.
2. column-parallel linear를 예로 들어 output feature 차원을 여러 rank로 나눌 때, 각 rank가 어떤 weight shard와 partial output을 갖는지 본다.
3. row-parallel linear를 예로 들어 input feature 차원을 나눌 때, 각 rank의 partial result를 왜 다시 합쳐야 하는지와 all-reduce/reduce-scatter 직관을 연결한다.
4. attention head, QKV projection, feed-forward expansion 같은 Transformer 내부 연산이 tensor parallel에 어떻게 잘 맞는지 관찰한다.
5. "메모리는 줄었는데 왜 step time은 반드시 좋아지지 않는가"를 질문하며 layer 내부 collective communication이 latency에 남기는 흔적을 읽는다.
6. 마지막에는 tensor parallelism을 ZeRO/FSDP 같은 sharding 접근, 다음 단위 `06_training_systems/06_pipeline_parallelism`, 이후 `06_training_systems/08_hybrid_parallel_topologies`와 연결해 본다.

## 이 단위에서 특히 볼 질문
- tensor parallelism은 DDP/FSDP/ZeRO처럼 상태를 나누는 접근과 무엇이 근본적으로 다른가?
- 왜 large Transformer에서는 레이어를 통째로 복제하는 것보다, 레이어 내부 matmul 차원을 나누는 편이 자연스러운 구간이 생기는가?
- column parallel과 row parallel은 각각 어떤 텐서 차원을 나누며, 어떤 시점에 collective communication이 필요한가?
- activation shard를 계속 쥔 채 다음 연산으로 넘길 수 있는 경우와, full activation을 다시 모아야 하는 경우는 어떻게 다른가?
- tensor parallelism은 메모리를 줄여 주는데도 왜 interconnect bandwidth와 latency에 민감한가?
- pipeline parallelism, sharding, hybrid parallelism을 볼 때 tensor parallel은 어떤 역할의 퍼즐 조각인가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 06_training_systems/05_tensor_parallelism/scratch_lab.py
{
  "status": "sample",
  "tp_world_size": 4,
  "input_shape": [8, 4096],
  "column_parallel_linear": {
    "global_weight_shape": [4096, 16384],
    "per_rank_weight_shape": [4096, 4096],
    "per_rank_output_shape": [8, 4096],
    "gather_output": true,
    "reconstructed_output_shape": [8, 16384]
  },
  "row_parallel_linear": {
    "global_weight_shape": [16384, 4096],
    "per_rank_input_shape": [8, 4096],
    "per_rank_partial_output_shape": [8, 4096],
    "collective": "all_reduce_sum"
  },
  "notes": "expected output/sample shape only"
}

$ python 06_training_systems/05_tensor_parallelism/framework_lab.py
{
  "status": "sample",
  "framework": "megatron-style tensor parallel",
  "attention_partition": {
    "num_heads_total": 32,
    "heads_per_rank": 8,
    "sequence_length": 2048,
    "hidden_size": 4096
  },
  "collectives_per_block": [
    "all_gather_or_reduce_scatter_on_activations",
    "all_reduce_on_partial_outputs"
  ],
  "observations": {
    "memory_per_rank": "smaller_than_full_layer_replica",
    "latency_risk": "communication_visible_every_layer",
    "preferred_topology": "high_bandwidth_intra-node_links"
  }
}
```

핵심은 숫자를 외우는 것이 아니라, **각 rank가 어떤 matrix shard와 activation shard를 들고 있는지**, **어느 시점에 full tensor를 다시 모아야 하는지**, **메모리 절감이 어떤 latency 비용과 함께 오는지**를 읽는 것이다.

## 다음 단위와의 연결
다음 단위 `06_training_systems/06_pipeline_parallelism`에서는 레이어 내부를 나누는 대신 **레이어 묶음을 stage로 쪼개는 방법**을 본다. 즉 tensor parallelism이 "한 레이어를 여러 GPU가 함께 계산하는 법"이라면, pipeline parallelism은 "서로 다른 레이어 구간을 여러 GPU가 순차로 맡는 법"을 다룬다.

그 다음 `06_training_systems/08_hybrid_parallel_topologies`에서는 data parallel + tensor parallel + pipeline parallel + sharding을 한 번에 묶는 현실적인 조합으로 넘어간다. 따라서 이 단위는 이후 분산 전략들을 비교할 때, **intra-layer split이라는 독립된 축**을 머릿속에 고정해 주는 연결 고리다.
