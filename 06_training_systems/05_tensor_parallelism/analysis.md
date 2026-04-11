# 05 Tensor Parallelism 분석

## Stable interpretation

Tensor parallelism is an intra-layer split: a single large layer is computed by
multiple ranks that each hold a matrix shard and an activation shard. This differs
from DDP-style replication and from FSDP/ZeRO-style state sharding because the
active matmul itself is partitioned.

## Korean-first reading

- column-parallel linear는 output feature 축을 나눠 rank별 activation shard를 만든다.
- row-parallel linear는 input feature 축과 weight row shard를 나눈 뒤 partial output을
  collective로 합친다.
- activation shard를 오래 유지하면 메모리와 bandwidth를 아낄 수 있지만, 다음
  연산이 같은 shard layout을 이해해야 한다.
- communication overhead는 메모리 절감의 대가다. 작은 CPU simulation에서도
  all-gather와 all-reduce가 어느 위치에 들어가는지 분리해 읽을 수 있다.

## Observed run

`analysis.py`는 `artifacts/scratch_metrics.json`과
`artifacts/framework_metrics.json`을 읽어 실행별 관측값을
`artifacts/analysis_observed.json`에 쓴다. 이 문서는 해석 프레임을 안정적으로
고정하기 위한 stable report이며, 실행별 숫자는 observed JSON을 확인한다.
