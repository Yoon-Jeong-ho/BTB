# 03 Sequence Models: RNN, LSTM, GRU 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, seq, feature)` 또는 `(seq, batch, feature)` 같은 시퀀스 텐서 shape 읽기
- hidden dimension과 output dimension이 같은 개념이 아니라는 점
- backpropagation과 gradient chain rule의 기본 감각
- token sequence, embedding, 시작/종료 토큰(BOS/EOS) 같은 NLP 기본 용어
- cross entropy가 "다음에 나올 정답 index"를 맞히는 데 자주 쓰인다는 점
- cosine similarity가 두 hidden state 방향 차이를 비교하는 도구라는 점
- PyTorch `RNN/LSTM/GRU` 가 CPU에서도 작은 toy batch를 충분히 재현할 수 있다는 점

## 빠른 자기 점검
- 같은 token 집합이라도 순서가 바뀌면 모델 출력이 달라져야 하는 예를 하나 들 수 있는가?
- hidden state와 final output의 역할 차이를 한두 문장으로 설명할 수 있는가?
- next-token prediction에서 입력 시퀀스와 정답 시퀀스를 왜 한 칸씩 shift하는지 설명할 수 있는가?
- gradient가 여러 시간 step을 거치며 약해질 수 있다는 말을 직관적으로 이해하는가?
- teacher forcing이 학습 시에는 도움이 되지만 추론 시 그대로 쓸 수는 없다는 점을 설명할 수 있는가?
