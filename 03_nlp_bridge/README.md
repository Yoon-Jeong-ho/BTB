# 02 NLP Bridge

이 구간은 `01_ml`에서 `03_nlp`로 넘어가기 전에 필요한 개념 다리다. 표형 데이터에서는 feature를 사람이 직접 설계했지만, NLP에서는 **문장을 토큰 id로 바꾸고 그 id를 embedding 공간으로 옮긴 뒤, 토큰끼리 attention으로 서로를 참고하게 만드는 과정**이 먼저 필요하다.

## 선행 / 다음 단계

- 선행 권장: [00_foundations](../00_foundations/README.md), 특히 `02_activation_and_loss`, `03_gradients_and_backpropagation`, `05_gpu_memory_runtime`
- 다음 단계: [03_nlp](../03_nlp/README.md)

## 브리지 내부 추천 순서

1. [01_tokenization_and_embeddings](01_tokenization_and_embeddings/README.md) — 문장이 subword 조각과 id sequence, embedding shape로 바뀌는 흐름을 본다.
2. [02_attention_and_transformer_block](02_attention_and_transformer_block/README.md) — attention weight, padding/causal mask, transformer block shape 흐름을 본다.

## 핵심 목표

- tokenization과 subword 분해가 왜 필요한지 이해한다.
- embedding lookup이 `정수 id -> dense vector` 변환이라는 사실을 shape로 확인한다.
- attention weight가 각 토큰의 정보를 어떻게 섞는지 작은 예제로 본다.
- padding mask와 causal mask가 왜 attention 계산의 안전장치인지 익힌다.
- `03_nlp`에 들어가기 전에 “문장이 모델 안에서 어떤 숫자 흐름으로 바뀌는가”를 설명할 수 있게 만든다.

## 각 unit에서 할 일

### 01_tokenization_and_embeddings

1. `scratch_lab.py`에서 작은 toy vocab으로 한국어 문장을 subword-ish하게 쪼개고 id로 바꾼다.
2. `framework_lab.py`에서 PyTorch `Embedding`으로 `(batch, seq)`가 `(batch, seq, dim)`으로 바뀌는 것을 확인한다.
3. `analysis.py`에서 관측치를 한국어로 해석하고, `THEORY.md`로 다시 연결한다.

### 02_attention_and_transformer_block

1. `scratch_lab.py`에서 손으로 만든 query/key/value로 attention score, softmax weight, weighted sum을 계산한다.
2. `framework_lab.py`에서 PyTorch attention과 transformer-block-style 연산으로 `(batch, seq, dim)`이 어떻게 유지되는지 확인한다.
3. `analysis.py`에서 sequence mixing, padding/causal mask, residual + feed-forward의 의미를 한국어로 정리한다.

## 학습 태도

- tokenizer를 “전처리 부속품”으로 보지 말고, **모델이 읽을 수 있는 단위로 문장을 재표현하는 규칙**으로 본다.
- 숫자 id 자체에는 의미가 없고, embedding lookup 이후에야 dense representation이 생긴다는 점을 계속 확인한다.
- attention output은 한 토큰의 복사가 아니라 여러 token value의 가중합이라는 점을 계속 의식한다.
- padding은 빈칸 채우기일 뿐 정보가 아니므로, mask 없이 평균/attention을 계산하면 왜 해석이 틀어지는지도 함께 본다.
