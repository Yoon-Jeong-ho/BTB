# 03 Machine Reading Comprehension 선행 개념

## 꼭 알고 오면 좋은 것
- Python 리스트, 딕셔너리, 반복문을 무리 없이 읽을 수 있는가?
- `(batch, seq_len)` 과 `(batch, seq_len, hidden_dim)` 같은 tensor shape를 읽을 수 있는가?
- exact match와 token F1이 서로 다른 질문에 답한다는 점을 이해하는가?
- `03_nlp_bridge`와 앞선 NLP task unit에서 tokenization, padding, embedding 기초를 이미 손으로 만져봤는가?

## 빠른 자기 점검
- 질문과 문맥을 한 시퀀스로 붙일 때 `[CLS]`, `[SEP]` 같은 special token이 왜 필요한지 설명할 수 있는가?
- 정답이 없는 질문을 그냥 아무 span으로 보내면 왜 위험한지 예시와 함께 말할 수 있는가?
- start / end 위치 예측과 answerable 분류를 동시에 본다는 말이 어떤 의미인지 감이 오는가?
