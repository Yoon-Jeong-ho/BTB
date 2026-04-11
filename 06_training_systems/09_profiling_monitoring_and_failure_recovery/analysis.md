# 09 Profiling, Monitoring, and Failure Recovery 분석

## Stable interpretation

Profiling is a short, deep view into a training window; monitoring is a long-running view of drift; recovery is the decision layer that uses those signals to choose whether to retry, resume, quarantine a checkpoint, or stop. This unit keeps all three together in a CPU-safe deterministic simulation so that the operational reasoning is testable without requiring GPUs.

## Korean-first reading

- profiling은 "느린 함수 찾기"가 아니라 step time을 data wait, forward/backward compute, communication wait, checkpoint I/O, misc sync로 나눠 보는 일이다.
- throughput 하락은 평균만 보면 늦게 보인다. p50/p95, tail latency, phase-boundary jitter를 함께 봐야 느려지는 구간을 빨리 찾을 수 있다.
- memory 관찰은 allocated, reserved, peak, phase를 분리해야 한다. reserved memory가 checkpoint 이후에도 높게 남으면 fragmentation이나 lifetime 문제를 의심한다.
- heartbeat와 per-rank 로그는 hang, slow step, straggler를 구분하는 liveness 신호다. 한 rank의 heartbeat lag가 collective wait 전체를 만들 수 있다.
- failure triage는 OOM, hang, divergence, storage/checkpoint failure를 먼저 나누고, 각 분류마다 첫 확인 지점을 다르게 둔다.
- recovery는 checkpoint 파일 유무가 아니라 manifest, optimizer_state, scheduler_state, sampler_state, RNG, global_step 연속성, post-resume validation까지 포함한 계약이다.

## Observed run

`analysis.py`는 `artifacts/scratch-manual/metrics.json`과 `artifacts/framework-manual/metrics.json`을 읽어 실행별 관측 보고서 `artifacts/analysis-manual/latest_report.md`를 쓴다. 이 문서는 stable report이며 실행별 숫자는 generated artifact에서 확인한다.

## 관련 이론

- [THEORY.md](./THEORY.md) — profiling / monitoring / recovery framing
- [PREREQS.md](./PREREQS.md) — 필요한 선행 감각
- 이전 단위 [06_pipeline_parallelism](../06_pipeline_parallelism/README.md) — step timeline과 communication wait 감각
