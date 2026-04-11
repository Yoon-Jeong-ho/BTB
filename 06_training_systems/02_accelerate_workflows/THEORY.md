# 02 Accelerate Workflows 이론 노트

## 핵심 개념
- Accelerate는 PyTorch training loop 위에 붙는 실행/장치 적응 계층이다.
- `Accelerator.prepare(...)`는 model, optimizer, dataloader, scheduler를 backend-aware wrapper로 감싼다고 이해할 수 있다.
- mixed precision과 device placement를 한곳에서 설정할 수 있지만, 수치 안정성과 backend semantics는 여전히 알아야 한다.

## 무엇을 줄여 주는가
- 반복적인 `.to(device)` 호출
- rank-aware launcher 설정 일부
- distributed dataloader 준비
- backward 호출 방식 차이 일부

## 무엇이 남는가
- batch/effective batch 해석
- overflow/underflow 같은 수치 문제
- checkpointing과 metric aggregation
- DeepSpeed/FSDP 같은 backend 자체의 trade-off

## Common Confusion
- Accelerate를 Trainer 같은 고수준 학습 프레임워크로 보는 실수
- `prepare()`를 호출하면 모든 distributed 개념을 몰라도 된다고 생각하는 실수
- mixed precision을 켜면 메모리와 속도가 항상 좋아진다고 믿는 실수

## 관찰 포인트
- baseline loop에서 사라진 device call은 몇 개인가?
- wrapper가 생겨도 사용자가 직접 해석해야 하는 값은 무엇인가?
- launch config와 실제 backend behavior는 어디서 갈라지는가?
