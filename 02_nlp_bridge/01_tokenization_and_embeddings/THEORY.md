# 01 Tokenization and Embeddings 이론 노트

## 핵심 개념
- **tokenization**은 원문 문자열을 모델 vocabulary 안의 단위로 바꾸는 과정이다.
- **subword splitting**은 단어 전체가 vocabulary에 없을 때 더 작은 조각으로 나눠 coverage를 확보하는 전략이다.
- **token id**는 vocabulary table에서 각 토큰에 붙인 정수 인덱스일 뿐, 자체 의미는 없다.
- **embedding lookup**은 `(batch, seq)` 정수 텐서를 `(batch, seq, hidden)` 실수 텐서로 바꾸는 table lookup이다.
- **padding mask**는 길이가 다른 sequence를 같은 배치로 묶을 때, 실제 토큰과 `[PAD]` 위치를 구분하는 표시다.

## 수식 / 직관
- 문자열 → 토큰: `"자연어를 좋아해요" -> ["자", "##연", "##어", "##를", "좋", "##아", "##해", "##요"]`
- 토큰 → id: `["[CLS]", "자", ... , "[SEP]"] -> [2, 4, ... , 3]`
- embedding lookup: `input_ids ∈ Z^(B×L) -> E[input_ids] ∈ R^(B×L×D)`
- padding mask: `mask = (input_ids != pad_id)` 이면 `mask.shape == (B, L)`

## Common Confusion
- token id 숫자 크기가 의미 크기라고 착각하는 실수
- whitespace 단어 수와 subword 토큰 수를 같은 길이 budget으로 생각하는 실수
- `[UNK]`가 하나 생기면 정보가 완전히 사라질 수 있다는 점을 놓치는 실수
- padding을 그냥 0 채움으로만 보고, average pooling이나 attention에서 mask가 왜 필요한지 잊는 실수
