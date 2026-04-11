# 03 DeepSpeed ZeRO 이론 노트

## 핵심 개념
- ZeRO는 data parallel에서 중복 저장되는 optimizer state, gradient, parameter를 단계적으로 나눠 갖는 전략이다.
- stage 1은 optimizer state, stage 2는 gradient, stage 3은 parameter까지 shard한다고 거칠게 이해할 수 있다.
- 메모리를 줄이는 대신 통신과 checkpoint 복구 복잡도가 늘어난다.

## 메모리 구성
- parameter: 모델 weight 자체
- gradient: backward에서 생기는 변화량
- optimizer state: Adam의 moment처럼 optimizer가 유지하는 추가 상태
- activation: forward 중 저장되는 중간 값

## Common Confusion
- ZeRO를 모델 구조 변경으로 오해하는 실수
- 메모리가 줄면 항상 빠르다고 믿는 실수
- stage 3이 언제나 최선이라고 단정하는 실수
- checkpoint format과 loading complexity를 무시하는 실수

## 관찰 포인트
- stage별 per-rank memory는 얼마나 줄어드는가?
- 어떤 상태가 어느 stage에서 처음 shard되는가?
- communication penalty가 어디서 커지는가?
