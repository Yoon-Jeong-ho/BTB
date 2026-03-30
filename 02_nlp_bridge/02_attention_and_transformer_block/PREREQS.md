# 02 Attention and Transformer Block 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, seq, dim)` 같은 텐서 shape 읽기
- dot product가 두 벡터 유사도를 수치로 만든다는 감각
- softmax가 점수를 확률 비슷한 weight로 바꾼다는 기본 이해
- `01_tokenization_and_embeddings`에서 embedding과 padding mask를 본 경험

## 빠른 자기 점검
- value들의 가중합이 "원래 토큰 하나"가 아니라 여러 토큰 정보의 혼합이라는 설명을 따라갈 수 있는가?
- padding mask가 없으면 `[PAD]` 위치가 attention에 섞여 들어갈 수 있다는 말을 이해하는가?
- transformer block이 shape를 유지하지만 내부 표현을 업데이트한다는 설명을 말로 풀 수 있는가?
