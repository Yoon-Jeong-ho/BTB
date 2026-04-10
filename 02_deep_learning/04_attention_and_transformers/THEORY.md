# 04 Attention and Transformers 이론 노트

## 핵심 개념

### 1. attention은 sequence mixing이다
- scaled dot-product attention은 `softmax(QK^T / sqrt(d_k))`로 각 query 위치가 어떤 key 위치를 얼마나 참고할지 weight를 만든다.
- 각 row 합이 1이라는 것은, query 위치 출력이 **value들의 가중합** 이라는 뜻이다.
- 그래서 attention output은 “원본 토큰 하나의 복사”가 아니라, **여러 위치 정보를 섞은 새 표현**으로 읽어야 한다.

### 2. multi-head는 여러 mixing 규칙을 병렬로 둔다
- head마다 projection이 다르면, 같은 시퀀스도 서로 다른 기준으로 읽는다.
- 어떤 head는 인접 문맥에, 다른 head는 긴 거리 dependency나 특정 역할 토큰에 더 큰 weight를 줄 수 있다.
- 즉 multi-head intuition의 핵심은 “한 번 크게 보는 것”이 아니라 **다른 관점의 mixing을 나란히 두는 것**이다.

### 3. encoder와 decoder는 attention을 쓰되 규칙이 다르다
- **encoder self-attention**은 보통 bidirectional이라 현재 위치가 왼쪽/오른쪽 문맥을 모두 볼 수 있다.
- **decoder self-attention**은 causal mask 때문에 미래 위치를 보면 안 된다.
- **encoder-decoder 구조**는 여기에 cross-attention이 더해져, decoder query가 encoder memory를 읽는다.
- 그래서 block 이름은 비슷해도 “누구를 볼 수 있는가”가 모델 패밀리 구분의 핵심이 된다.

### 4. transformer는 recurrent bottleneck을 완화한다
- RNN/LSTM/GRU는 hidden state를 시간축으로 순서대로 넘겨야 해서, 긴 의존성일수록 정보 경로가 길어진다.
- self-attention은 한 layer 안에서 필요한 위치를 직접 참조할 수 있어, 멀리 떨어진 토큰 사이 경로를 훨씬 짧게 만든다.
- 학습 시에는 위치별 상호작용을 한 번에 계산할 수 있어 병렬화가 쉽다.
- 대신 attention matrix는 시퀀스 길이 증가에 따라 `O(seq^2)` 비용을 가진다. bottleneck이 사라진 것이 아니라 **형태가 바뀐 것**이다.

### 5. 이 단위는 model family를 읽는 기준을 만든다
- encoder-only: 전체 입력을 bidirectional self-attention으로 읽는 분류/이해 계열
- decoder-only: causal self-attention으로 다음 토큰을 예측하는 autoregressive 계열
- encoder-decoder: encoder memory + decoder cross-attention으로 입력/출력을 나누는 seq2seq 계열
- 이 기준이 있으면 BERT / GPT / T5 류를 이름 외우기보다 구조로 분류하게 된다.

## Common Confusion
- attention weight가 크면 “그 토큰 하나만 복사했다”고 오해하는 실수
- multi-head를 단순 파라미터 증가로만 보고, head별 관점 차이를 놓치는 실수
- encoder와 decoder가 둘 다 self-attention을 쓰므로 사실상 같은 블록이라고 착각하는 실수
- transformer가 recurrent 병목을 줄이므로 길이 비용 문제까지 모두 해결했다고 생각하는 실수
- cross-attention을 “decoder self-attention의 다른 이름”으로 오해하는 실수

## 실행에서 꼭 확인할 것
- scratch 실험에서 각 attention row 합이 1에 가깝게 유지되는가?
- head별 top key가 달라져, multi-head가 서로 다른 mixing 관점을 만든다는 신호가 보이는가?
- encoder 규칙에서는 미래 위치 weight가 남고, decoder 규칙에서는 causal mask 때문에 미래 weight가 0으로 막히는가?
- framework 실험에서 encoder hidden / decoder hidden / decoder block output shape가 모두 `(batch, seq, dim)`으로 유지되는가?
- recurrent 경로 길이와 attention 경로 길이를 비교하면 어떤 병목 완화가 드러나는가?

## 실행 결과 예시
```text
scratch metrics
- sequence_length: 5
- max_row_sum_error: 0.0
- multi_head.head_count: 2
- encoder_decoder.encoder_future_access_mass: 0.465313
- encoder_decoder.causal_mask_future_blocked: true
- figure_path: artifacts/scratch-manual/attention_patterns.svg

framework metrics
- device: cpu
- num_heads: 2
- encoder_hidden_shape: [2, 5, 8]
- decoder_hidden_shape: [2, 5, 8]
- cross_attention_used: true
- encoder_future_attention_mean: 0.18210457
- decoder_future_attention_max: 0.0
```
이 숫자는 “attention row sum은 mixing 비율”, “multi-head는 여러 관점”, “decoder는 미래 차단”, “transformer는 recurrence 없이 병렬 mixing”이라는 이론 문장을 실제 관측값으로 다시 붙여 준다.
