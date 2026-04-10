# 01 Language Modeling and Pretraining Objectives 선행 개념

## 꼭 알고 오면 좋은 것
- token, vocabulary, logit, cross entropy의 기본 감각
- autoregressive next-token prediction이 무엇인지
- `[MASK]`나 sentinel token 같은 corruption marker가 왜 필요한지
- encoder-only / decoder-only / encoder-decoder 구분
- context window가 “볼 수 있는 범위”를 제한한다는 점
- objective와 architecture가 완전히 같은 말이 아니라는 점

## 먼저 다시 보면 좋은 단위
- [03_nlp_bridge/01_tokenization_and_embeddings](../../03_nlp_bridge/01_tokenization_and_embeddings/README.md)
- [03_nlp_bridge/02_attention_and_transformer_block](../../03_nlp_bridge/02_attention_and_transformer_block/README.md)
- [02_deep_learning/04_attention_and_transformers](../../02_deep_learning/04_attention_and_transformers/README.md)
- [05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture](../02_corpus_tokenizer_and_data_mixture/README.md)

## 빠른 자기 점검
- causal LM에서 입력과 타깃을 한 칸 shift하는 이유를 설명할 수 있는가?
- masked LM에서 loss-mask density가 낮아지는 이유를 설명할 수 있는가?
- span corruption에서 sentinel token이 왜 필요한지 말할 수 있는가?
- context window와 long-term memory를 같은 말로 보면 왜 문제가 되는지 설명할 수 있는가?
- 같은 transformer family여도 objective가 바뀌면 model behavior intuition이 달라진다는 말을 받아들일 수 있는가?

## 다음에 이어서 보면 좋은 단위
- [02_corpus_tokenizer_and_data_mixture](../02_corpus_tokenizer_and_data_mixture/README.md) — tokenizer/data mixture 설계가 objective를 어떻게 지지하는지 후속 연결로 확인한다.
