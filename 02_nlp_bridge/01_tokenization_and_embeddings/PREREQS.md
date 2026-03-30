# 01 Tokenization and Embeddings 선행 개념

## 꼭 알고 오면 좋은 것
- 문자열과 리스트를 구분해 읽는 기본 Python 감각
- 인덱스(index)와 table lookup의 개념
- 텐서 shape `(batch, seq, hidden)`를 읽는 기본 감각

## 빠른 자기 점검
- 같은 문장이라도 whitespace 단어 수와 subword token 수가 달라질 수 있다는 말을 이해하는가?
- 정수 id 자체에는 의미가 없고 embedding table을 거쳐야 dense vector가 생긴다는 설명을 따라갈 수 있는가?
- 길이가 다른 두 문장을 같은 batch로 묶으려면 왜 `[PAD]`와 mask가 필요한지 말할 수 있는가?
