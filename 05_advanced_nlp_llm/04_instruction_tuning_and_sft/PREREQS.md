# 04 Instruction Tuning and SFT 선행 개념

## 꼭 알고 오면 좋은 것
- causal LM과 next-token loss가 기본적으로 어떻게 동작하는지에 대한 감각
- prompt, response, template, EOS 같은 생성 인터페이스 기본 용어
- train / validation split과 labeled supervision 데이터셋의 기본 구조
- role/message 형식(system, user, assistant)이 대화형 모델 입력으로 쓰인다는 점
- fine-tuning이 base model의 행동을 이동시키되 모든 능력을 새로 만들지는 않는다는 점
- format adherence와 factual correctness를 구분해서 봐야 한다는 점

## 빠른 자기 점검
- instruction tuning이 pretraining과 완전히 별개라기보다 instruction-formatted example 위의 supervised next-token 학습이라는 설명을 받아들일 수 있는가?
- plain instruction-response format과 chat template가 모델에 다른 conditioning signal을 준다는 점을 설명할 수 있는가?
- system message가 user 질문과 다른 역할을 하는 이유를 한두 문장으로 말할 수 있는가?
- assistant 답변 구간에 loss를 집중하는 설정이 왜 자주 쓰이는지 설명할 수 있는가?
- SFT로 말투와 형식은 정렬돼도 preference ranking 문제는 별도로 남는다는 점을 이해하는가?

## 먼저 다시 보면 좋은 단위
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — next-token objective와 supervision target framing 복습
- [02_corpus_tokenizer_and_data_mixture](../02_corpus_tokenizer_and_data_mixture/README.md) — instruction 데이터도 결국 corpus/format 설계라는 관점 연결
- [04_nlp/03_machine_reading_comprehension](../../04_nlp/03_machine_reading_comprehension/README.md) — 입력-정답 framing이 모델 행동을 바꾼다는 감각 복습
