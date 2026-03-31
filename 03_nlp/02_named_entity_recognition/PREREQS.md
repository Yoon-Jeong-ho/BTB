# 02 Named Entity Recognition 선행 개념

## 꼭 알고 오면 좋은 것
- Python 리스트, 딕셔너리, 반복문을 무리 없이 읽을 수 있는가?
- `(batch, seq_len, hidden_dim)` 같은 sequence tensor shape를 읽을 수 있는가?
- `B-LOC`, `I-LOC`, `O` 같은 BIO tag를 보고 entity span을 복원할 수 있는가?
- `02_nlp_bridge`에서 tokenization, padding, embedding 기초를 이미 손으로 만져봤는가?

## 빠른 자기 점검
- word-level label을 subword token으로 늘려 붙이는 이유를 설명할 수 있는가?
- token accuracy와 entity-level F1이 서로 다른 질문에 답한다는 점을 이해하는가?
- sequence labeling에서 앞뒤 문맥이 왜 중요한지 예시와 함께 말할 수 있는가?
