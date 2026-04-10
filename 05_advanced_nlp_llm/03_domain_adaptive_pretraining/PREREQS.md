# 03 Domain Adaptive Pretraining 선행 개념

## 꼭 알고 오면 좋은 것
- causal LM / masked LM / span corruption 같은 pretraining objective가 무엇을 유지한 채 적응하는지에 대한 기본 감각
- corpus quality, tokenizer, mixture 설계가 실제 token stream과 학습 신호를 바꾼다는 점
- validation loss, perplexity, held-out set이 모델 변화를 추적하는 기본 도구라는 점
- domain shift가 vocabulary 차이만이 아니라 문체, 형식, 길이, 정보 밀도 차이까지 포함한다는 점
- fine-tuning과 continued pretraining이 같은 적응이 아니라는 점
- catastrophic forgetting이 새 데이터 적응의 반대편 비용으로 나타날 수 있다는 점

## 빠른 자기 점검
- "같은 objective를 유지한 채 데이터를 바꿔서 계속 pretrain한다"는 말이 왜 DAPT의 핵심인지 설명할 수 있는가?
- pure-domain continued pretraining이 빠른 specialization과 forgetting 위험을 동시에 만들 수 있는 이유를 말할 수 있는가?
- in-domain validation 하나만 보고 stop 시점을 정하면 어떤 문제가 생길지 예를 들 수 있는가?
- domain corpus selection에서 양보다 품질/중복/오염/최신성이 먼저 중요해질 수 있는 상황을 떠올릴 수 있는가?
- DAPT와 instruction tuning이 각각 무엇을 바꾸는 단계인지 한두 문장으로 구분할 수 있는가?

## 먼저 다시 보면 좋은 단위
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — 어떤 objective를 유지한 채 적응하는지 먼저 정리한다.
- [02_corpus_tokenizer_and_data_mixture](../02_corpus_tokenizer_and_data_mixture/README.md) — 어떤 domain corpus를 얼마나 어떤 mixture로 더 넣을지 설계 관점을 복습한다.
- [04_nlp/01_text_classification](../../04_nlp/01_text_classification/README.md) — in-domain downstream evaluation을 읽는 최소 감각을 연결한다.
