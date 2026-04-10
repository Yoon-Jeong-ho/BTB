# 01 Language Modeling and Pretraining Objectives 선행 개념

## 꼭 알고 오면 좋은 것
- token, vocabulary, logit, cross entropy가 각각 무엇을 뜻하는지에 대한 기본 감각
- sequence length와 context window가 모델 입력 범위를 제한한다는 점
- autoregressive prediction이 "이전 토큰들로 다음 토큰을 맞히는 문제" 라는 점
- encoder-only / decoder-only / encoder-decoder 구조를 아주 거칠게라도 구분할 수 있다는 점
- self-attention이 문맥을 읽는 기본 방식과 positional information의 필요성
- 학습 objective와 추론 사용 방식이 완전히 같은 말은 아니라는 점

## 먼저 다시 보면 좋은 단위
- [03_nlp_bridge/01_tokenization_and_embeddings](../../03_nlp_bridge/01_tokenization_and_embeddings/README.md) — token과 subword, embedding 복습
- [03_nlp_bridge/02_attention_and_transformer_block](../../03_nlp_bridge/02_attention_and_transformer_block/README.md) — self-attention이 문맥을 읽는 방식 복습
- [02_deep_learning/04_attention_and_transformers](../../02_deep_learning/04_attention_and_transformers/README.md) — transformer block과 context 처리 직관 복습
- [04_nlp/03_machine_reading_comprehension](../../04_nlp/03_machine_reading_comprehension/README.md) — 입력/정답 framing이 task behavior를 바꾼다는 감각 연결

## 빠른 자기 점검
- causal LM에서 입력과 정답을 한 칸씩 shift하는 이유를 설명할 수 있는가?
- masked LM이 loss를 전체 위치가 아니라 일부 mask 위치에만 거는 이유를 말할 수 있는가?
- span corruption이 token 하나 예측과 어떻게 다른지 sentinel token 예시로 설명할 수 있는가?
- context window는 "볼 수 있는 범위" 와 "기억을 잘 쓰는 능력" 이 같은 말이 아니라는 점을 받아들일 수 있는가?
- 같은 transformer 계열 구조라도 objective가 달라지면 학습 신호와 downstream 강점이 달라진다는 말을 이해하는가?
