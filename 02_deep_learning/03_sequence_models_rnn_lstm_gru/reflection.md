# 03 Sequence Models 회고 질문

- 오늘 본 예시에서 **순서가 바뀌었을 뿐인데 final hidden state가 달라진 이유**를 자신의 말로 설명해 보자.
- vanilla RNN의 `long-range signal`이 gated unit보다 작게 남았다면, 그것이 곧 어떤 학습 실패(예: 긴 문맥 기억 실패)로 이어질지 적어 보자.
- LSTM의 forget gate와 GRU의 update gate를 각각 "무엇을 유지하려는 장치"로 이해했는지 한 문장씩 써 보자.
- teacher forcing loss는 낮은데 free-running loss가 더 큰 상황을 보며, **학습 중 정답 이전 토큰을 보는 것**과 **추론 중 자기 예측을 다시 넣는 것**의 차이를 어떻게 느꼈는가?
- 다음 단위의 attention은 hidden state 병목을 어떤 방식으로 완화하려는지, 지금 단위와 연결해 짧게 적어 보자.
