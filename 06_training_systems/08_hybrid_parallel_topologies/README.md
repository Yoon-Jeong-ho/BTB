# 08 Hybrid Parallel Topologies

> Status: runnable
>
> 이 단위는 GPU를 직접 요구하지 않는 **CPU-safe deterministic topology simulation**이다. 실제 분산 런타임을 띄우지 않고도 data parallel, tensor parallel, pipeline parallel, FSDP/state sharding 축을 한 장의 device mesh 설계 문제로 묶어 읽는다.

## 왜 이 단위를 배우는가

`06_training_systems/04_fsdp_checkpointing_and_offload`, `05_tensor_parallelism`, `06_pipeline_parallelism`, `07_data_parallel_grad_accumulation`까지 오면 이제 중요한 질문은 각 기법을 따로 설명하는 것이 아니라, **실제 대형 모델 학습에서 이 축들을 어떻게 함께 묶을 것인가**다. 현실의 LLM/멀티모달 모델 학습은 data parallel만으로도, tensor parallel만으로도, pipeline parallel만으로도, FSDP만으로도 끝나지 않는다. 모델 상태 메모리, activation 메모리, intra-layer 연산량, stage bubble, global batch 운영, checkpoint 복구 제약이 동시에 얽히기 때문이다.

그래서 hybrid parallel topology는 단순한 "옵션 조합"이 아니라, **모델 규모와 하드웨어 배치를 연결하는 설계 문제**로 봐야 한다. 어느 축을 node 안에 둘지, 어느 축을 node 사이에 둘지, 통신이 잦은 축을 빠른 interconnect 위에 올릴지, memory-saving 축과 throughput 축을 어떤 비율로 섞을지 결정하는 감각이 필요하다. 이 단위는 여러 parallelism 축을 한 장의 topology 그림으로 합쳐 읽고, 그 선택이 memory fit, communication tradeoff, bottleneck reasoning, checkpoint portability에 어떤 흔적을 남기는지 관찰한다.

## 실행 순서

```bash
python 06_training_systems/08_hybrid_parallel_topologies/scratch_lab.py
python 06_training_systems/08_hybrid_parallel_topologies/framework_lab.py
python 06_training_systems/08_hybrid_parallel_topologies/analysis.py
```

세 스크립트는 모두 deterministic CPU simulation이다.

- `scratch_lab.py`는 64-GPU planning case에서 여러 candidate topology를 만들고, `DP x TP x PP` axis product, FSDP shard factor, memory budget, communication hotspot, bottleneck reasoning을 계산한다.
- `framework_lab.py`는 같은 후보를 더 framework-like scoring으로 읽어 rank mesh contract, fast/slow link 배치, checkpoint portability signal을 정리한다.
- `analysis.py`는 두 metrics 파일이 존재해야 실행되며, stable report(`analysis.md`)와 observed JSON을 남긴다.

## 생성되는 산출물

- `artifacts/scratch_metrics.json` — scratch topology planner의 후보별 memory/communication budget
- `artifacts/hybrid_topology_mesh.svg` — selected topology의 DP/TP/PP/FSDP 축 관계 그림
- `artifacts/framework_metrics.json` — framework-style topology scoring과 rank mesh contract
- `artifacts/analysis_observed.json` — 실행별 관측 요약
- `analysis.md` — 안정적인 해석 프레임

## 이번 단위에서 특히 볼 축

### Data parallel axis

Data parallel은 replica와 batch 축이다. global/effective batch, gradient synchronization cadence, optimizer step timing을 책임진다. Hybrid topology에서는 보통 바깥 축으로 두기 쉽지만, DP group이 커질수록 gradient all-reduce 또는 reduce-scatter 비용이 무시되지 않는다.

### Tensor parallel axis

Tensor parallel은 레이어 내부 matmul/attention head split이다. `tp_all_reduce`, `all_gather` 같은 collective가 block마다 자주 등장하기 때문에 latency-sensitive하다. 이 단위의 preferred candidate는 TP4를 node-local fast link 위에 두는 선택을 보여 준다. 같은 world size라도 tensor-parallel group이 느린 node 간 link를 타면 step time bottleneck이 먼저 커질 수 있다.

### Pipeline parallel axis

Pipeline parallel은 layer stack을 stage로 나누고 microbatch schedule을 만든다. 모델 residency를 줄이는 데 도움이 되지만 stage boundary activation `send/recv`, bubble fraction, load balance 위험을 남긴다. PP depth를 늘리면 per-stage memory는 줄 수 있지만, microbatch 수와 stage imbalance가 충분히 관리되지 않으면 throughput이 떨어진다.

### FSDP / state sharding axis

FSDP/ZeRO류 sharding은 parameter, gradient, optimizer state의 resident memory와 lifecycle을 바꾼다. 이는 compute split이라기보다 state placement와 checkpoint contract 문제에 가깝다. Hybrid topology에서는 어떤 DP/FSDP group 기준으로 shard를 저장하고, 어떤 TP/PP mesh metadata를 함께 남겨야 reload/restart가 안전한지가 중요해진다.

## 실행 결과 예시

`scratch_lab.py`는 다음 형태의 JSON을 출력한다. 숫자는 CPU-only planning model의 deterministic 추정값이며 실제 네트워크 벤치마크가 아니다.

```json
{
  "status": "runnable",
  "simulation": "deterministic_cpu_hybrid_topology_planner",
  "cpu_safe_simulation": true,
  "parallel_axes": {
    "data_parallel": "replica / batch axis and gradient synchronization cadence",
    "tensor_parallel": "intra-layer matrix and attention-head split; latency-sensitive collectives",
    "pipeline_parallel": "layer-stage split plus microbatch time-axis schedule",
    "fsdp_state_sharding": "parameter/gradient/optimizer state residency and checkpoint lifecycle"
  },
  "preferred_candidate": "tp4_pp2_dp8_fsdp_hybrid",
  "selection_summary": {
    "axis_product": "DP8 x TP4 x PP2",
    "primary_risk": "pipeline_or_fsdp_overlap",
    "reason": [
      "keeps tensor-parallel collectives inside fast node-local links",
      "uses pipeline depth 2 to reduce model residency without excessive bubble",
      "keeps FSDP/state sharding as an explicit checkpoint-aware memory axis"
    ]
  }
}
```

`framework_lab.py`는 같은 후보를 scoring 관점에서 다시 읽는다.

```json
{
  "status": "runnable",
  "framework": "deterministic_cpu_hybrid_parallel_topology_sim",
  "device_mesh_axes": [
    "data_parallel",
    "tensor_parallel",
    "pipeline_parallel",
    "fsdp_state_sharding"
  ],
  "rank_mesh_contract": {
    "rank_order": "dp_outer / pp_middle / tp_inner",
    "tp_inner_reason": "tensor-parallel all-reduce/all-gather is latency-sensitive, so keep it inside fast node-local links",
    "pp_middle_reason": "pipeline stages can cross node boundaries when activation payload and bubble are budgeted",
    "dp_fsdp_outer_reason": "data replica and FSDP shard groups define batch cadence, state residency, and checkpoint remap contract"
  }
}
```

## 해석 방법

핵심은 특정 숫자를 외우는 것이 아니라, **모델 규모를 어떤 병렬화 축 조합으로 하드웨어에 끼워 넣는지**, **그때 통신과 메모리 부담이 어느 경계로 이동하는지**, **운영 가능한 checkpoint / batch / schedule 계약이 무엇인지**를 읽는 것이다.

좋은 답은 보통 다음 문장을 모두 포함한다.

- “이 topology의 world size는 DP, TP, PP 축의 곱으로 맞는다.”
- “하지만 world size가 맞는 것만으로는 부족하고, tensor-parallel traffic이 어떤 link를 타는지가 중요하다.”
- “FSDP는 memory fit을 돕지만 all-gather peak와 checkpoint metadata 계약을 남긴다.”
- “Pipeline depth는 residency를 줄이지만 bubble/load balance와 activation transfer를 만든다.”
- “Data parallel 축은 batch budget과 optimizer cadence를 정한다.”

## 다음 단위와의 연결

다음 단위 `06_training_systems/09_profiling_monitoring_and_failure_recovery`에서는 여기서 설계한 hybrid topology가 실제로 돌 때, 어느 링크가 막히는지, 어느 stage가 놀고 있는지, 어떤 rank에서 OOM이나 timeout이 먼저 나는지, checkpoint 복구가 왜 꼬이는지 같은 **운영 흔적**을 본격적으로 다룬다.

즉 이 단위가 topology를 종이 위에서 설계하는 단계라면, 다음 단위는 그 설계가 실제 runtime에서 어떤 로그·metric·failure mode로 나타나는지 읽는 단계다. 여기서 parallel axes를 한 장의 시스템 그림으로 묶어 두면, profiling과 recovery를 볼 때도 “이 병목이 tensor parallel 때문인지, pipeline bubble 때문인지, FSDP shard lifecycle 때문인지”를 더 정확히 분해할 수 있다.
