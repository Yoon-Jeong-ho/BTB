# 04 Attention and Transformers 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, seq, dim)` 형태의 텐서 shape를 자연스럽게 읽는 습관
- dot product, matrix multiplication, softmax가 attention score를 weight로 바꾼다는 기본 감각
- `02_deep_learning/03_sequence_models_rnn_lstm_gru`에서 본 recurrent hidden state 업데이트와 그 병목
- `03_nlp_bridge/02_attention_and_transformer_block`에서 본 self-attention, padding mask, causal mask 기초
- residual connection, layer normalization, feed-forward block이 "shape는 유지하고 표현은 바꾼다"는 기본 이해

## 빠른 자기 점검
- attention output이 왜 "토큰 하나의 복사"가 아니라 value들의 가중합인지 설명할 수 있는가?
- multi-head를 두는 이유를 "그냥 더 크게 만들기 위해서"보다 조금 더 정확하게 말할 수 있는가?
- encoder는 전체 문맥을 보고 decoder는 미래를 막는다는 차이를 설명할 수 있는가?
- transformer가 recurrent bottleneck을 줄여 준다는 말과, 동시에 attention cost가 늘 수 있다는 말을 함께 이해하는가?
- BERT류 / GPT류 / seq2seq transformer를 각각 encoder-only / decoder-only / encoder-decoder로 분류할 준비가 되어 있는가?
