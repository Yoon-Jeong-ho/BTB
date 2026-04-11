# 09 Profiling, Monitoring, and Failure Recovery Reflection

## 실행 전 예측

1. `scratch_lab.py`를 실행하기 전에 어떤 signal이 먼저 움직이면 throughput 하락을 가장 빨리 설명할 수 있을지 예측하라.
2. 같은 step time 증가라도 compute hotspot, communication wait, checkpoint I/O 중 어떤 경우에 recovery decision이 달라질지 적어 보라.
3. memory peak가 allocated보다 reserved에서 더 오래 남는다면 OOM triage에서 어떤 추가 질문을 던질지 적어 보라.

## 실행 후 기록할 것

1. `step_time_ms.p50`과 `step_time_ms.p95`를 비교하고, average step time 하나만 봤을 때 놓칠 수 있는 tail latency를 설명하라.
2. `time_breakdown_pct.communication_wait`와 `per_rank_heartbeat`를 함께 보며 왜 `rank_2_heartbeat_lag`가 병목 가설의 첫 split이 되는지 정리하라.
3. `memory_snapshot.peak_reserved_mb`, `peak_allocated_mb`, `peak_phase`를 사용해 OOM 위험이 batch-size 문제인지 checkpoint/eval boundary 문제인지 분류하라.
4. `framework_lab.py`의 monitoring contract에서 throughput, step time, loss/grad norm, gpu memory, per-rank heartbeat, checkpoint freshness가 모두 필요한 이유를 한 문단으로 설명하라.
5. failure taxonomy에서 OOM, hang_or_straggler, divergence_after_resume, storage_checkpoint_failure가 각각 요구하는 first check를 비교하라.
6. `recovery_decision.action`이 `retry_from_last_good_checkpoint`인 이유를 checkpoint manifest와 post-resume validation 관점에서 설명하라.
7. retry가 안전하지 않은 failure class를 하나 골라, checkpoint quarantine 또는 human escalation이 필요한 이유를 적어 보라.

## 자기 점검

- 나는 profiling을 step time / memory / communication timeline으로 설명할 수 있다.
- 나는 monitoring snapshot이 profiler trace보다 길게 drift를 보는 장치라는 점을 설명할 수 있다.
- 나는 OOM, hang, divergence를 같은 failure로 뭉뚱그리지 않고 첫 질문을 다르게 던질 수 있다.
- 나는 recoverable checkpoint에 optimizer_state, scheduler_state, sampler_state, RNG, global_step, manifest가 필요한 이유를 말할 수 있다.
- 나는 recovery decision이 retry policy와 post-resume validation 없이 자동화되면 위험하다는 점을 설명할 수 있다.
