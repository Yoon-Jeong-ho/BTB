# 06 Pipeline Parallelism Reflection

## 실행 후 기록할 것

1. `scratch_lab.py`의 `partition_plan`에서 각 pipeline stage가 맡는 layer range와
   `compute_units`를 적어 보라. 레이어 수가 비슷해도 partition balance가 완전히
   같지 않은 이유는 무엇인가?
2. `schedule_trace`를 보고 warmup, steady, cooldown 구간을 직접 표시하라. 어느
   slot이 pipeline bubble로 남는가?
3. microbatch 수 6개와 stage 3개에서 `bubble_fraction`이 왜 0.25로 계산되는지
   `idle_stage_slots / total_stage_slots` 형태로 설명하라.
4. `activation_transfer.boundary_payload_elements`를 보며 stage boundary마다 어떤
   activation transfer 계약(shape/dtype/order)이 필요할지 적어 보라.
5. `framework_lab.py`의 1F1B simulation에서 GPipe peak saved microbatch 수와
   `one_f1b_peak_saved_microbatches`를 비교하라. throughput과 activation memory
   trade-off를 본인의 말로 정리하라.
6. tensor parallelism은 레이어 내부 split, pipeline parallelism은 pipeline stage
   split, data parallelism은 replica/batch split이라는 차이를 later hybrid
   parallelism 관점에서 한 문단으로 정리하라.

## 자기 점검

- 나는 pipeline parallelism을 "여러 GPU 사용"이 아니라 execution path를 stage로
  나누는 방식이라고 설명할 수 있다.
- 나는 microbatch schedule이 warmup / steady / cooldown과 bubble을 만든다는 것을
  시간축으로 그릴 수 있다.
- 나는 activation transfer가 stage boundary의 shape, dtype, ordering 계약이라는
  점을 설명할 수 있다.
- 나는 partition balance를 레이어 개수가 아니라 compute / memory / communication
  payload를 함께 보는 문제로 이해한다.
