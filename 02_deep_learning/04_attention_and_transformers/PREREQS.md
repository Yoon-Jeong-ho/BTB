# 04 Attention and Transformers 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, seq, dim)` 텐서 shape를 자연스럽게 읽는 습관
- dot product, matrix multiplication, softmax가 attention score를 weight로 바꾸는 기본 감각
- `02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 본 recurrent hidden state 업데이트와 long-range bottleneck
- `02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 본 recurrent bottleneck과 sequence modeling 감각
- residual connection, layer normalization, feed-forward block이 shape는 유지하고 표현은 바꾼다는 이해

## 빠른 자기 점검
- attention output이 왜 value들의 가중합인지 설명할 수 있는가?
- attention row sum이 1이라는 사실이 왜 중요한지 말할 수 있는가?
- multi-head가 “한 개 큰 head”와 직관적으로 어떻게 다른지 설명할 수 있는가?
- encoder와 decoder의 가장 큰 차이를 **미래 토큰 접근 가능 여부**로 설명할 수 있는가?
- transformer가 recurrent bottleneck을 줄여도, 길이 제곱 비용이 남는 이유를 이해하는가?
- encoder-only / decoder-only / encoder-decoder 모델을 구조 기준으로 분류할 준비가 되어 있는가?
