# 05 Tensor Parallelism Reflection

## 실행 후 기록할 것

1. `scratch_lab.py`에서 column-parallel linear의 행렬 shard shape와 activation shard
   shape를 적어 보라. 이 split은 왜 output feature 축을 나눈다고 말할 수 있는가?
2. row-parallel linear에서 각 rank의 partial output이 dense output과 같아지려면
   어떤 collective가 필요한가? `all_reduce_sum`이 들어가는 이유를 한 문장으로
   설명하라.
3. `framework_lab.py`의 attention partition에서 rank당 head 수는 얼마인가? 이
   model parallel split이 batch split과 어떻게 다른지 비교하라.
4. `communication_overhead.estimated_bytes`와
   `throughput_model.communication_share`를 보며, tensor parallelism이 메모리를
   줄여도 communication overhead 때문에 step time이 늘 수 있는 상황을 가정하라.
5. FSDP는 상태 sharding, tensor parallelism은 intra-layer split, pipeline
   parallelism은 stage split이라는 차이를 본인의 말로 정리하라.

## 자기 점검

- 나는 column-parallel과 row-parallel의 split 축을 구분할 수 있다.
- 나는 activation shard를 유지하는 설계와 full activation을 다시 모으는 설계의
  trade-off를 설명할 수 있다.
- 나는 tensor parallelism을 FSDP/pipeline parallelism과 섞어 쓰는 이유를 "서로
  다른 병렬화 축"이라는 관점으로 설명할 수 있다.
