# 07 Data Parallel + Grad Accumulation Reflection

## 실행 전 예측

1. local batch 8, world size 4, grad accumulation 4에서 global batch와 effective batch가 각각 얼마인지 먼저 계산하라.
2. every-step all-reduce와 deferred sync/no_sync를 비교했을 때 communication 횟수가 어느 쪽에서 줄어드는지 예측하라.
3. loss normalization 없이 microstep loss를 그대로 backward하면 gradient scale이 어떤 방향으로 틀어질지 적어 보라.
4. gradient clipping을 microstep마다 할 때와 accumulation boundary 뒤에 할 때 optimizer step cadence 해석이 어떻게 달라지는지 예측하라.

## 실행 후 관찰

1. `scratch_lab.py`의 `accumulation_trace`에서 optimizer step이 발생하는 microstep을 표시하라.
2. `sync_policy_comparison`에서 every-step sync와 deferred sync의 all-reduce 횟수 차이를 계산하고, no_sync가 무엇을 미루는지 설명하라.
3. `memory_model_mb`를 보며 큰 local batch와 작은 local batch + grad accumulation 중 activation memory peak가 낮은 쪽을 고르라.
4. `framework_lab.py`의 `throughput_model`을 보고 effective batch가 커져도 wall-clock throughput이 자동 개선되지 않는 이유를 한 문장으로 정리하라.
5. scheduler step이 microstep 기준이 아니라 optimizer step 기준이어야 하는 상황을 본인의 실험 로그 예시로 써 보라.

## 다음 단위 연결

- data parallel은 batch 축, tensor parallel은 layer 내부 연산 축, pipeline parallel은 stage/time 축이라는 차이를 한 문단으로 비교하라.
- hybrid parallel topology에서 data parallel 축이 batch budget과 optimizer step cadence를 계속 책임지는 이유를 설명하라.
