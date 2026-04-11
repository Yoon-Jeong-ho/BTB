# 04 FSDP Checkpointing and Offload 성찰 질문

## 실행 전 예측
1. DDP full replica와 FSDP shard resident state 중 어떤 값이 per-rank memory 기준선이 되는가?
2. activation checkpointing을 켜면 peak memory와 step time이 각각 어느 방향으로 움직일지 먼저 예측해 보라.
3. CPU offload를 켰을 때 “더 빨라진다”가 아니라 “살아남는다”에 가까운 이유를 적어 보라.

## 실행 후 관찰
1. `scratch_lab.py`의 SVG에서 가장 큰 memory drop은 어떤 전환에서 나타났는가?
2. `framework_lab.py`에서 full state dict와 sharded state dict의 load peak 차이는 resume 전략을 어떻게 바꾸는가?
3. `analysis-manual/latest_report.md`의 실행 조치 중 실제 대형 학습 장애 대응 runbook에 넣고 싶은 항목은 무엇인가?

## 다음 단위 연결
1. FSDP sharding은 “상태”를 나누고, tensor parallelism은 “연산”을 나눈다. 두 문장을 자신만의 예시로 구분해 보라.
2. hybrid parallel topology에서 FSDP, tensor parallel, pipeline parallel을 동시에 쓰면 checkpoint 형식 선택이 왜 더 중요해지는가?
