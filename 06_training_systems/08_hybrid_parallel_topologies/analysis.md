# 08 Hybrid Parallel Topologies 분석

## Stable interpretation

Hybrid parallel topology planning is a model-hardware placement problem, not a
checklist that turns on every parallelism option. Data parallelism owns the
replica/batch axis, tensor parallelism owns latency-sensitive intra-layer
collectives, pipeline parallelism owns stage/time scheduling, and FSDP/ZeRO-style
state sharding owns state residency plus checkpoint lifecycle.

## Korean-first reading

- data parallel 축은 global/effective batch와 gradient sync cadence를 담당한다.
- tensor parallel 축은 레이어 내부 matmul/head split을 만들며, all-reduce/all-gather가
  자주 발생하므로 빠른 node-local link 위에 두는 편이 안전하다.
- pipeline parallel 축은 stage boundary와 microbatch schedule을 만들고, activation
  send/recv와 bubble/load-balance 위험을 남긴다.
- FSDP/state sharding 축은 parameter, gradient, optimizer state의 resident memory와
  checkpoint save/load 계약을 바꾼다.
- 좋은 hybrid topology는 memory fit, communication tradeoff, bottleneck reasoning,
  checkpoint portability를 동시에 읽을 수 있어야 한다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
