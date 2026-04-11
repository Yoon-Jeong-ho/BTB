# 03 DeepSpeed ZeRO 선행 개념

## 꼭 알고 오면 좋은 것
- DDP가 모델 복사본마다 같은 parameter를 들고 있다는 점
- optimizer state / gradient / parameter가 서로 다른 메모리 항목이라는 점
- `06_training_systems/01_torchrun_and_ddp_basics`와 `02_accelerate_workflows`의 실행 계약

## 빠른 자기 점검
- Adam optimizer state가 왜 추가 메모리를 쓰는지 설명할 수 있는가?
- stage가 올라가면 memory와 communication이 동시에 바뀐다는 점을 이해하는가?
- checkpoint를 저장/불러올 때 shard가 왜 문제가 될 수 있는가?
