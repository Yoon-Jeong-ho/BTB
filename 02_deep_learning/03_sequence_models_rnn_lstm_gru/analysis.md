# 03 Sequence Models 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 조금 바뀌어도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- recurrent model의 핵심은 `h_t = f(x_t, h_{t-1})` 라는 누적 규칙이다. 그래서 같은 token 집합이어도 순서가 바뀌면 hidden trajectory와 final state가 달라진다.
- vanilla RNN은 초반 신호를 hidden state 하나에 계속 접어 넣는 구조라 long-range dependency에서 빠르게 희석되기 쉽다. `rnn_long_range_signal`이 작아지는 방향으로 읽는다.
- LSTM/GRU는 gate를 통해 "얼마나 유지할지 / 얼마나 덮어쓸지"를 따로 조절한다. 따라서 `lstm_long_range_signal`, `gru_long_range_signal`이 더 크게 남는다면 gated memory가 초기 단서를 더 오래 붙잡았다고 해석할 수 있다.
- teacher forcing은 학습을 안정화하지만, 추론에서는 정답 이전 토큰 대신 모델 자신의 예측을 넣어야 한다. 그래서 `teacher_forcing_loss` 와 `free_running_loss` 사이 gap을 exposure bias 관점에서 읽는다.
- 이 해석이 바로 다음 attention 단위의 출발점이다. "왜 모든 과거를 hidden state 하나에 눌러 담아야 하는가?" 라는 질문이 self-attention의 동기를 만든다.

## 확인 질문
- 같은 원소 집합인데도 순서만 바뀌면 왜 final hidden state가 달라지는가?
- 이번 실행에서 vanilla RNN과 LSTM/GRU의 long-range signal 차이는 얼마나 컸는가?
- teacher forcing gap이 생겼다면, 그것이 train/infer mismatch를 어떻게 보여 주는가?
- hidden bottleneck을 이해한 뒤 attention을 보면 어떤 문제가 더 선명하게 보이는가?

## 관련 이론
- [THEORY.md](./THEORY.md): hidden state, gating, teacher forcing, transformer 연결을 다시 확인한다.
