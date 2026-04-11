# 01 Torchrun and DDP Basics 선행 개념

## 꼭 알고 오면 좋은 것
- mini-batch와 gradient가 무엇인지
- optimizer step이 parameter를 gradient 방향으로 움직인다는 점
- CPU/GPU device가 코드 실행 위치를 바꿀 수 있다는 점
- `02_deep_learning/07_training_recipes_and_debugging`에서 본 batch / gradient / logging 감각

## 빠른 자기 점검
- 같은 모델을 여러 rank에 복사한다는 말을 이해하는가?
- gradient 평균이 왜 필요한지 한 문장으로 말할 수 있는가?
- global rank와 local rank를 구분할 수 있는가?
