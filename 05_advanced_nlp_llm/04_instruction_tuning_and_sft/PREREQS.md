# 04 Instruction Tuning and SFT 선행 개념

## 꼭 알고 오면 좋은 것
- causal LM과 next-token loss가 serialized sequence에서 어떻게 계산되는지에 대한 기본 감각
- prompt, response, template, EOS, label, loss mask 같은 생성/학습 인터페이스 용어
- train/validation split과 labeled supervision dataset의 기본 구조
- role/message 형식(system, user, assistant)이 chat model 입력으로 쓰인다는 점
- fine-tuning이 base model의 모든 능력을 새로 만들기보다 행동 분포를 이동시킨다는 점
- format adherence, factual correctness, helpfulness를 서로 구분해서 봐야 한다는 점

## 빠른 자기 점검
- instruction format이 단순한 문자열 장식이 아니라 모델이 보는 input-output template라는 점을 설명할 수 있는가?
- supervised fine-tuning이 reference assistant answer를 모방하게 만드는 단계라는 점을 이해하는가?
- system/user/assistant role framing이 conditioning signal로 작동한다는 말을 받아들일 수 있는가?
- assistant-only loss mask가 prompt 복창보다 response 생성을 강조하는 이유를 말할 수 있는가?
- imitation과 helpfulness가 같은 것이 아니며, preference optimization이 별도로 필요한 이유를 예로 들 수 있는가?

## 먼저 다시 보면 좋은 단위
- [01_language_modeling_and_pretraining_objectives](../01_language_modeling_and_pretraining_objectives/README.md) — next-token objective와 target framing 복습
- [02_corpus_tokenizer_and_data_mixture](../02_corpus_tokenizer_and_data_mixture/README.md) — instruction dataset도 결국 corpus/template 설계라는 관점 연결
- [03_domain_adaptive_pretraining](../03_domain_adaptive_pretraining/README.md) — base LM 분포 적응과 assistant behavior 적응을 분리해서 이해한다.
