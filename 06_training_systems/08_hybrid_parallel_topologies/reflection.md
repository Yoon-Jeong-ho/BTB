# 08 Hybrid Parallel Topologies Reflection

## 실행 후 기록할 것

1. `scratch_lab.py`의 `parallel_axes`를 보고 data parallel, tensor parallel,
   pipeline parallel, FSDP/state sharding이 각각 무엇을 나누는지 한 문장씩 적어 보라.
2. preferred candidate의 `axis_product`를 확인하고 `DP x TP x PP` 곱이 왜 world size
   계약이 되는지 설명하라. 단, 곱이 맞는 것만으로 topology 설계가 끝나지 않는 이유도 함께 적어라.
3. `communication_budget.communication_hotspots`에서 가장 latency-sensitive한 축은 무엇인가?
   tensor-parallel collectives를 node-local fast link에 두려는 이유를 본인의 말로 정리하라.
4. `memory_budget`에서 FSDP shard factor, tensor parallel, pipeline stage split이 per-rank
   memory를 어떻게 함께 줄이는지 구분하라. resident memory와 peak memory를 섞지 말라.
5. `framework_lab.py`의 `rank_mesh_contract`를 보고 rank order가 checkpoint portability와
   failure recovery에 어떤 metadata를 남겨야 하는지 적어 보라.
6. 다음 단위 profiling에서 먼저 확인할 signal 세 가지를 고르라. 예: `tp_all_reduce` latency,
   pipeline bubble, FSDP all-gather peak, checkpoint remap failure.

## 자기 점검

- 나는 hybrid parallelism을 “병렬화 옵션을 모두 켜기”가 아니라 모델-하드웨어 배치 설계 문제로 설명할 수 있다.
- 나는 data/tensor/pipeline/FSDP 축이 서로 다른 병목을 완화한다는 점을 구분할 수 있다.
- 나는 통신이 많은 축을 빠른 링크에 배치해야 하는 이유와, 잘못 배치했을 때 어떤 bottleneck이 생기는지 말할 수 있다.
- 나는 memory fit, throughput, checkpoint portability, implementation complexity가 동시에 최적화되기 어렵다는 점을 받아들인다.
