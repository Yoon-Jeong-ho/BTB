# 02 Attention and Transformer Block 이론 노트

## 핵심 개념
- **self-attention**은 각 토큰이 같은 시퀀스 안의 다른 토큰을 얼마나 참고할지 weight로 계산하는 메커니즘이다.
- **query / key / value**는 각각 "무엇을 찾는가", "무엇을 제공하는가", "실제로 섞을 정보는 무엇인가"를 담당하는 표현이다.
- attention output은 각 query 위치에서 value들의 **가중합(weighted sum)** 으로 만들어진다.
- **padding mask**는 `[PAD]` 같은 빈 위치를 참고하지 못하게 막는다.
- **causal mask**는 decoder처럼 미래 토큰을 보면 안 되는 상황에서 오른쪽 미래 위치를 가린다.
- **transformer block**은 보통 `attention -> residual -> layer norm -> feed-forward -> residual -> layer norm` 흐름으로 hidden state를 업데이트한다.

## 수식 / 직관
- score: `scores = QK^T / sqrt(d_k)`
- weight: `A = softmax(scores)`
- output: `context = A @ V`
- shape 흐름:
  - input hidden states: `(batch, seq, dim)`
  - attention weights: `(batch, heads, query_seq, key_seq)`
  - attention output: `(batch, seq, dim)`
  - transformer block output: `(batch, seq, dim)`

## Common Confusion
- attention weight를 "확률 해석만 가능한 분류값"으로 오해하는 실수
- softmax 이후 row 합이 1이라는 사실과, 그래서 output이 value들의 convex combination이라는 점을 놓치는 실수
- padding mask는 key 쪽을 가리는 장치인데 query 위치까지 자동으로 지워준다고 착각하는 실수
- transformer block이 shape를 보존하므로 "아무 변화가 없다"고 생각하는 실수

## 이 단위에서 꼭 확인할 것
- 특정 query 토큰이 어떤 key 토큰으로 가장 크게 weight를 주는가?
- mask를 걸면 어느 열(column)이 0으로 사라지는가?
- residual connection이 attention output을 더해도 왜 최종 shape는 그대로인가?
