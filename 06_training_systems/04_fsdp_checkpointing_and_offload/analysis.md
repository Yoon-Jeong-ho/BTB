# 04 FSDP Checkpointing and Offload 분석

## 해석 프레임
- 이 파일은 안정적인 분석 가이드다. `analysis.py`를 실행해도 이 파일은 바뀌지 않고, 실행 관측은 `artifacts/analysis-manual/latest_report.md`에 기록된다.
- FSDP는 parameter/gradient/optimizer state를 rank별 shard로 두고, 계산 직전에 필요한 full parameter view를 잠깐 all-gather하는 runtime으로 읽는다.
- activation checkpointing은 모델 저장용 checkpoint가 아니라 학습 중 activation 저장량을 줄이는 recomputation 전략이다.
- CPU offload는 GPU memory pressure를 낮추지만 transfer latency와 jitter를 새 비용으로 만든다.
- full state dict와 sharded state dict는 같은 모델을 저장하더라도 resume/export/debug 계약이 다르다.

## 읽어야 할 숫자
- `ddp_full_replica_per_rank_mb`: 각 rank가 전체 상태를 들 때의 기준선이다.
- `fsdp_forward_peak_gpu_mb`: all-gather로 full parameter view가 materialize되는 순간의 peak다.
- `fsdp_checkpointed_peak_gpu_mb`: activation checkpointing으로 activation resident memory를 줄인 뒤의 peak다.
- `cpu_offload_gpu_peak_mb`: optimizer shard를 CPU로 내렸을 때 GPU 기준 peak다.
- `full_state_dict.load_peak_mb`와 `sharded_state_dict.load_peak_mb`: checkpoint 형식이 load-time memory를 어떻게 바꾸는지 보여 준다.

## 확인 질문
- 왜 FSDP steady-state memory와 forward peak memory를 따로 봐야 하는가?
- activation checkpointing이 줄이는 memory와 늘리는 compute는 각각 무엇인가?
- CPU offload는 어떤 상황에서 합리적이고, 어떤 상황에서 iteration 속도를 해치는가?
- full state dict는 언제 유리하고 sharded state dict는 언제 유리한가?
- tensor parallelism을 배우기 전에 FSDP가 해결하는 문제가 “상태 배치”라는 점을 어떻게 설명할 수 있는가?
