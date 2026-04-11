# 06 Pipeline Parallelism 분석

## Stable interpretation

Pipeline parallelism is execution-path partitioning: a sequential model is split
into pipeline stages, and microbatches move across those stages over time. It is
not a real multi-device runtime in this unit; the labs use deterministic CPU
simulation to make schedule, bubble, throughput, activation transfer, and
partition-balance trade-offs visible.

## Korean-first reading

- pipeline stage는 레이어 묶음을 맡는 실행 구간이며, stage boundary는 activation
  transfer 계약을 만든다.
- microbatch schedule은 warmup / steady / cooldown을 만들고, fill/drain 구간의
  idle slot이 pipeline bubble로 관찰된다.
- microbatch 수가 늘면 bubble fraction은 줄어들 수 있지만, transfer 횟수와
  bookkeeping도 함께 늘어난다.
- 1F1B는 backward를 더 빨리 시작해 GPipe식 all-forward-then-all-backward보다
  activation 보관량을 낮추는 방향의 schedule policy다.
- partition은 레이어 개수 균등 분할이 아니라 stage별 compute, memory,
  communication payload를 함께 맞추는 문제다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
