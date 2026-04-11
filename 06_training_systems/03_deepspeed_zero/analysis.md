# 03 DeepSpeed ZeRO 분석

## 해석 프레임
- ZeRO는 data parallel 중복 상태를 shard해 per-rank memory를 줄인다.
- stage가 올라갈수록 memory는 줄지만 communication/checkpoint complexity는 커진다.
- 이 단위의 숫자는 실제 DeepSpeed 실행이 아니라 memory accounting intuition이다.
