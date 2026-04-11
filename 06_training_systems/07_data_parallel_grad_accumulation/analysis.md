# 07 Data Parallel + Grad Accumulation 분석

## Stable interpretation

Data parallelism expands the batch axis: every rank keeps a full model replica,
processes a different data shard, and then participates in gradient all-reduce.
Grad accumulation changes optimizer step cadence: several microsteps contribute
to one optimizer update, so effective batch is local batch × world size ×
accumulation steps.

## Korean-first reading

- data parallel은 모델을 복제하고 batch shard를 나누는 축이다. 모델 내부 연산을
  쪼개는 tensor parallel이나 stage를 나누는 pipeline parallel과 구분한다.
- grad accumulation은 local batch를 그대로 둔 채 optimizer step을 늦춰 effective
  batch를 키우는 스케줄링 정책이다.
- accumulation window 안에서는 `no_sync`/deferred sync로 all-reduce 횟수를 줄일 수
  있지만, boundary에서는 여전히 gradient synchronization 계약을 지켜야 한다.
- loss normalization은 microstep loss를 accumulation step 수로 나눠 gradient scale을
  맞추는 장치다.
- gradient clipping과 scheduler step은 microstep마다가 아니라 accumulation boundary의
  optimizer step 직전에/직후에 해석해야 한다.
- 같은 effective batch라도 큰 local batch와 작은 local batch + accumulation은
  activation memory, kernel efficiency, throughput trace가 다르게 보인다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 안정 해석 프레임이며, 실행별
숫자는 observed JSON을 확인한다.
