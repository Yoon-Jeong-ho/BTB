# 06 Decoder Generation Bridge

이 문서는 `04_nlp`의 encoder/task-head 감각에서 `05_advanced_nlp_llm`의 decoder-only LLM 감각으로 넘어가기 위한 짧은 다리다. 텍스트 분류나 NER처럼 입력 전체를 보고 label을 고르는 문제와, LLM처럼 **지금까지 만든 token을 다시 입력으로 넣어 다음 token을 고르는 문제**는 실행 흐름이 다르다.

## 왜 필요한가

`04_nlp`까지의 많은 실습은 encoder representation 위에 task head를 붙인다. 반면 LLM 실습은 prompt를 token sequence로 만들고, 매 step마다 logits에서 다음 token을 고른 뒤, 그 token을 다시 context에 붙인다. 이 차이를 모르면 SFT, preference optimization, RLHF에서 “policy가 답변을 만든다”는 문장이 추상적으로 남는다.

## 모델 taxonomy 먼저 정리하기

| 모델 형태 | 대표 감각 | 주로 하는 일 | mask / head 관점 |
| --- | --- | --- | --- |
| **encoder-only** | 입력 전체를 양방향으로 읽어 representation을 만든다 | 분류, NER, retrieval embedding | padding mask + task head |
| **decoder-only** | 지금까지 본 token만 보고 다음 token을 예측한다 | chat, completion, code generation | causal mask + LM head |
| **encoder-decoder** | 입력을 encoder로 조건화하고 decoder가 출력 sequence를 만든다 | 번역, 요약, captioning | encoder padding mask + decoder causal mask |

LLM 단원에서 “causal LM”이라고 하면 보통 decoder-only 모델을 뜻한다. 이때 학습 label은 입력 token과 한 칸 어긋난 next-token target이다. 예를 들어 `나는 밥을`을 넣고 마지막 위치에서 `먹었다`의 확률을 높이는 식이다. 그래서 `04_nlp`의 classification head와 달리, `05_advanced_nlp_llm`에서는 **LM head가 매 token 위치마다 vocabulary logits를 만든다**는 점을 먼저 잡아야 한다.

## Autoregressive decoding loop

실무에서는 이 과정을 **autoregressive** generation이라고 부른다.

1. prompt를 token id로 바꾼다.
2. model이 현재 context의 다음 token logits를 낸다.
3. decoding rule이 logits에서 하나의 token을 고른다.
4. 고른 token을 context 뒤에 붙인다.
5. stop token 또는 max length까지 2~4를 반복한다.

핵심은 정답 label을 한 번에 맞히는 것이 아니라, **자기 출력이 다음 입력이 되는 반복 과정**이라는 점이다.

## Greedy vs sampling

- **greedy decoding**: 매 step에서 확률이 가장 높은 token만 고른다. 재현성은 높지만 답변이 단조롭거나 조기 고정될 수 있다.
- **sampling**: 확률분포에서 token을 뽑는다. 다양성은 생기지만 품질과 안전성이 흔들릴 수 있다.
- **beam search**: 여러 후보 sequence를 동시에 유지한다. 번역처럼 정답 문장 탐색이 중요한 문제에는 유용하지만 chat-style 생성에서는 과도하게 딱딱해질 수 있다.

## Temperature / top-k / top-p

- **temperature**: logits 분포를 평평하게 하거나 날카롭게 만든다. 높이면 다양한 token이 나오고, 낮추면 높은 확률 token에 더 집중한다.
- **top-k**: 확률이 높은 k개 token 안에서만 sampling한다.
- **top-p**: 누적 확률 p에 들어오는 token 집합 안에서 sampling한다. 상황마다 후보 수가 달라질 수 있다.

실험에서는 같은 prompt를 두고 decoding 설정만 바꿔 길이, 반복, hallucination, 정답성 변화를 비교한다.

## Prompt serialization

LLM은 사람이 보는 “질문/답변” 구조를 raw string으로 직접 이해하지 않는다. 실제 입력은 다음과 같은 serialized text다.

```text
<system>너는 친절한 튜터다.</system>
<user>attention을 설명해줘.</user>
<assistant>
```

SFT나 preference dataset을 볼 때는 “내용”뿐 아니라 role token, separator, answer boundary가 어떻게 들어갔는지 확인해야 한다.

## KV-cache intuition

매 step마다 전체 prefix를 다시 계산하면 느리다. decoder transformer는 이전 token들의 key/value를 cache하고, 새 token에 대해서만 추가 계산한다. 그래서 inference에서는 다음을 같이 본다.

- context length가 길어질수록 cache memory가 늘어난다.
- batch size와 max new tokens가 latency와 memory를 동시에 흔든다.
- 긴 답변은 품질 문제가 아니라 systems 문제가 될 수도 있다.

## 다음 단원에서 확인할 질문

- `05_advanced_nlp_llm/01_language_modeling_and_pretraining_objectives`에서 causal LM은 어떤 token 위치를 loss로 계산하는가?
- `05_advanced_nlp_llm/04_instruction_tuning_and_sft`에서 prompt/answer boundary는 어디인가?
- `05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto`에서 chosen/rejected 답변은 같은 prompt serialization을 공유하는가?

## 최소 실험 아이디어

- 같은 prompt를 greedy, temperature 0.7, top-p 0.9로 각각 5회 생성한다.
- 답변 길이, 반복 phrase, 사실 오류, instruction following 실패를 표로 남긴다.
- 긴 prompt와 짧은 prompt에서 KV-cache memory가 어떻게 달라질지 추정한다.
