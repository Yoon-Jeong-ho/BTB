# 02 NLP Bridge

이 구간은 `01_ml`에서 `03_nlp`로 넘어가기 전에 필요한 개념 다리다. 표형 데이터에서는 feature를 사람이 직접 설계했지만, NLP에서는 **문장을 토큰 id로 바꾸고 그 id를 embedding 공간으로 옮긴 뒤, 토큰끼리 attention으로 서로를 참고하게 만드는 과정**이 먼저 필요하다. 이 bridge는 그 전환을 한국어 예제로 천천히 연결한다.

## 핵심 목표

- tokenization과 subword 분해가 왜 필요한지 이해한다.
- embedding lookup이 `정수 id -> dense vector` 변환이라는 사실을 shape로 확인한다.
- attention weight가 각 토큰의 정보를 어떻게 섞는지 작은 예제로 본다.
- padding mask와 causal mask가 왜 attention 계산의 안전장치인지 익힌다.
- `03_nlp`에 들어가기 전에 “문장이 모델 안에서 어떤 숫자 흐름으로 바뀌는가”를 설명할 수 있게 만든다.

## 브리지 단위 구성

| Unit | 다루는 질문 | 남길 산출물 |
| --- | --- | --- |
| [01_tokenization_and_embeddings](01_tokenization_and_embeddings/README.md) | 한국어 문장이 어떻게 subword 조각과 id sequence가 되고, embedding/padding mask shape로 이어지는가? | scratch metrics, framework metrics, 한국어 analysis |
| [02_attention_and_transformer_block](02_attention_and_transformer_block/README.md) | embedding된 토큰들은 attention weight로 서로를 어떻게 섞고, transformer block 안에서 mask와 residual을 거치며 어떤 shape를 유지하는가? | scratch attention metrics, framework attention metrics, 한국어 analysis |

### 01_tokenization_and_embeddings에서 할 일

1. `scratch_lab.py`에서 작은 toy vocab으로 한국어 문장을 subword-ish하게 쪼개고 id로 바꾼다.
2. `framework_lab.py`에서 PyTorch `Embedding`으로 `(batch, seq)`가 `(batch, seq, dim)`으로 바뀌는 것을 확인한다.
3. `analysis.py`에서 관측치를 한국어로 해석하고, `THEORY.md`로 다시 연결한다.

### 02_attention_and_transformer_block에서 할 일

1. `scratch_lab.py`에서 손으로 만든 query/key/value로 attention score, softmax weight, weighted sum을 계산한다.
2. `framework_lab.py`에서 PyTorch attention과 transformer-block-style 연산으로 `(batch, seq, dim)`이 어떻게 유지되는지 확인한다.
3. `analysis.py`에서 sequence mixing, padding/causal mask, residual + feed-forward의 의미를 한국어로 정리한다.

## 학습 태도

- tokenizer를 “전처리 부속품”으로 보지 말고, **모델이 읽을 수 있는 단위로 문장을 재표현하는 규칙**으로 본다.
- 숫자 id 자체에는 의미가 없고, embedding lookup 이후에야 dense representation이 생긴다는 점을 계속 확인한다.
- attention output은 한 토큰의 복사가 아니라 여러 token value의 가중합이라는 점을 계속 의식한다.
- padding은 빈칸 채우기일 뿐 정보가 아니므로, mask 없이 평균/attention을 계산하면 왜 해석이 틀어지는지도 함께 본다.

## 다음 단계와의 연결

이 bridge를 끝내면 `03_nlp`의 텍스트 분류, NER, MRC에서 보게 될 tokenizer / attention mask / embedding layer / transformer encoder block을 더 이상 추상 용어로 보지 않게 된다.
