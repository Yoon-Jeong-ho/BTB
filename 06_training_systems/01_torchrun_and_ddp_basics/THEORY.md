# 01 Torchrun and DDP Basics 이론 노트

## 핵심 개념
- **torchrun**은 여러 Python worker를 띄우고 각 worker에 `RANK`, `LOCAL_RANK`, `WORLD_SIZE` 같은 실행 정보를 준다.
- **rank**는 전체 프로세스 중 몇 번째인지, **local rank**는 한 노드 안에서 몇 번째 device/process인지 나타낸다.
- **DDP(Distributed Data Parallel)**는 모델 전체를 각 rank에 복사하고, 각 rank가 다른 mini-batch를 본 뒤 gradient를 평균내 같은 update를 적용한다.

## single process와 distributed launch 차이
- single process는 한 코드 흐름에서 한 batch를 보고 한 gradient를 만든다.
- DDP는 여러 rank가 각자 batch slice를 보고 gradient를 만든 뒤 평균을 맞춘다.
- 따라서 핵심은 “모델을 쪼갔다”가 아니라 “같은 모델 복사본들이 서로 다른 데이터 조각을 본 뒤 gradient를 동기화한다”는 점이다.

## Common Confusion
- DDP를 model parallel과 혼동하는 실수
- local rank를 global rank와 같은 말로 보는 실수
- world size가 커지면 자동으로 학습이 좋아진다고 믿는 실수
- gradient accumulation과 DDP all-reduce를 같은 단계로 보는 실수

## 관찰 포인트
- rank별 gradient가 서로 다른가?
- 평균 gradient가 모든 rank에 같은 update로 적용되는가?
- local rank mapping이 device assignment와 어떻게 연결되는가?
