# 03 Sequence Models: RNN, LSTM, GRU 이론 노트

## 핵심 개념

### 1. 순서가 있는 입력은 왜 별도로 다뤄야 하는가
- 문장, 로그, 시계열처럼 순서가 중요한 데이터는 같은 원소 집합이라도 배치 순서가 바뀌면 의미가 달라진다.
- MLP나 bag-of-features 관점은 전체 입력을 한 번에 읽기 때문에, "먼저 왔는가 나중에 왔는가"를 직접 표현하기 어렵다.
- recurrent model은 입력을 한 step씩 읽으며 과거 요약본을 다음 step으로 넘긴다.

### 2. hidden state는 무엇을 하는가
- 기본 recurrent update는 `h_t = f(x_t, h_{t-1})` 형태로 쓸 수 있다.
- 여기서 `h_t`는 t시점까지 본 정보를 압축한 요약 상태다.
- 같은 가중치를 모든 시간 step에서 재사용하므로, 모델은 "같은 읽기 규칙"으로 시퀀스를 훑는다.
- 장점은 순서와 길이를 다룰 수 있다는 것이고, 단점은 많은 정보를 작은 state에 계속 눌러 담아야 한다는 병목이다.

### 3. vanilla RNN의 한계
- vanilla RNN은 긴 시퀀스에서 멀리 떨어진 정보가 약해지기 쉽다.
- 이유는 time unrolling 뒤 backpropagation through time(BPTT)을 하면 비슷한 변환이 여러 번 곱해지고, 이 과정에서 gradient가 점점 작아지거나 커질 수 있기 때문이다.
- 그래서 long-range dependency, 예를 들어 문장 앞부분의 주어 정보가 뒤쪽 동사 해석에 오래 남아야 하는 문제에서 불안정해지기 쉽다.
- 또 모든 과거 정보를 사실상 hidden state 하나에 접어 넣어야 하므로, 정보 압축 손실이 쉽게 생긴다.

### 4. LSTM의 gating intuition
- LSTM은 hidden state 외에 **cell state**라는 비교적 직접적인 memory 통로를 둔다.
- forget gate는 기존 memory를 얼마나 지울지, input gate는 새 정보를 얼마나 쓸지, output gate는 현재 memory를 얼마나 밖으로 드러낼지 조절한다.
- 직관적으로는 "무조건 모두 기억"이 아니라 "기억/삭제/노출 비율을 학습"하는 구조다.
- 그래서 vanilla RNN보다 긴 문맥을 더 안정적으로 버티는 경우가 많지만, 모든 장기 의존성을 완벽하게 해결하는 마법은 아니다.

### 5. GRU의 gating intuition
- GRU는 LSTM보다 단순한 구조로, 보통 update gate와 reset gate를 중심으로 설명한다.
- update gate는 이전 상태를 얼마나 유지할지, reset gate는 과거 정보를 새 candidate state 계산에서 얼마나 덜 볼지를 조절한다.
- LSTM보다 state 구조가 간단해 구현/해석이 조금 덜 복잡한 편이지만, 핵심 아이디어는 여전히 "정보 흐름을 gate로 조절한다"는 데 있다.

### 6. teacher forcing과 sequence modeling setup
- next-token prediction에서는 보통 입력 시퀀스와 정답 시퀀스를 한 칸씩 어긋나게 만든다.
  - 입력: `[BOS, 나, 는, 학생]`
  - 정답: `[나, 는, 학생, EOS]`
- teacher forcing은 학습 중 이전 시점의 **정답 토큰** 을 다음 step 입력으로 넣어주는 방식이다.
- 이렇게 하면 학습이 더 안정적이고 빠를 수 있지만, 추론 시에는 정답 대신 **모델 자신의 예측** 을 넣어야 하므로 train/infer mismatch(exposure bias)가 생긴다.

### 7. transformer로 가는 다리
- recurrent family는 과거 정보를 hidden state 경로로 순차 전달한다.
- 반면 transformer는 attention으로 필요한 과거 위치를 직접 참조하려고 한다.
- 그래서 RNN/LSTM/GRU를 배우는 목적 중 하나는, transformer가 무엇을 개선하려는지 선명하게 이해하는 것이다.
- 질문은 간단하다. "왜 모든 과거를 `h_t` 하나로 압축해야 하는가?" 이 질문이 다음 단위의 출발점이다.

## Common Confusion
- hidden state를 "과거 전체의 lossless 저장본"으로 오해하는 실수
- LSTM/GRU가 있으면 long-context 문제가 완전히 사라진다고 생각하는 실수
- gate를 0 아니면 1의 스위치처럼 상상하는 실수
- teacher forcing을 "정답을 미리 보여주는 부정행위"로 이해하는 실수
- final hidden state만 중요하고, 중간 step의 state 변화는 안 봐도 된다고 넘기는 실수
- `(seq, batch, hidden)`와 `(batch, seq, hidden)` shape convention을 섞어 읽는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 같은 token 집합이라도 순서를 바꾸면 hidden state trajectory가 달라지는가?
- 멀리 떨어진 첫 토큰의 신호가 vanilla RNN에서는 얼마나 빨리 약해지는가?
- LSTM의 forget/input gate 또는 GRU의 update gate가 특정 패턴에서 유지/갱신 쪽으로 어떻게 기울어지는가?
- teacher forcing loss와 free-running loss 사이에 어느 정도 간격이 생기는가?
- next-token setup에서 입력과 타깃이 정확히 한 칸씩 shift되었는가?
- 마지막으로, attention은 왜 "한 hidden state에 과거를 접는 방식" 대신 "필요한 위치를 다시 꺼내 보는 방식"으로 이해할 수 있는가?
