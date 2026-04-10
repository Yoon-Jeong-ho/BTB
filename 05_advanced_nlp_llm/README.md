# 05 Advanced NLP + LLM

이 트랙은 `03_nlp_bridge`와 `04_nlp` 다음에 들어오는 **pretraining부터 post-training까지 포괄하는 advanced NLP+LLM** 구간이다. 기본 NLP 태스크를 한 번 돌려 본 뒤, 이제는 언어모델 objective, tokenizer/data mixture, domain adaptive pretraining, instruction tuning, preference optimization, RLHF, RAG, alignment를 한 흐름으로 묶어 본다.

즉 `03 → 04 → 05` 구간에서 `작은 태스크 실습`을 `대형 언어모델을 만들고 적응시키고 평가하는 파이프라인`으로 확장하는 역할을 맡는다. `06_training_systems`가 분산 시스템이라면, 이 트랙은 모델/데이터/정렬 관점의 고급 NLP·LLM 설계에 초점을 둔다.

현재는 `01~08` 전 unit가 outlined 문서 단계까지 정리돼 있다. 즉 이 트랙은 전체 흐름을 문서와 메타데이터 기준으로 먼저 따라가며 이해할 수 있고, 이후 runnable lab는 우선순위에 따라 순차적으로 채워질 예정이다.

## 단위 구성

| Unit | Status | Focus |
| --- | --- | --- |
| [01_language_modeling_and_pretraining_objectives](01_language_modeling_and_pretraining_objectives/README.md) | outlined | causal / masked / span corruption objective가 어떤 inductive bias를 만드는지 정리한다. |
| [02_corpus_tokenizer_and_data_mixture](02_corpus_tokenizer_and_data_mixture/README.md) | outlined | tokenizer와 corpus mixture가 학습 분포와 budget을 어떻게 바꾸는지 본다. |
| [03_domain_adaptive_pretraining](03_domain_adaptive_pretraining/README.md) | outlined | 기존 LM을 특정 도메인으로 계속 pretrain할 때의 전략과 trade-off를 다룬다. |
| [04_instruction_tuning_and_sft](04_instruction_tuning_and_sft/README.md) | outlined | instruction format과 supervised fine-tuning이 사용성에 어떻게 연결되는지 본다. |
| [05_preference_optimization_dpo_orpo_kto](05_preference_optimization_dpo_orpo_kto/README.md) | outlined | 선호 데이터로 정책을 직접 업데이트하는 post-training objective를 비교한다. |
| [06_rlhf_and_reasoning_rl](06_rlhf_and_reasoning_rl/README.md) | outlined | reward model과 policy optimization이 reasoning behavior를 어떻게 바꾸는지 익힌다. |
| [07_retrieval_augmented_generation_and_eval](07_retrieval_augmented_generation_and_eval/README.md) | outlined | RAG 파이프라인과 평가 harness를 함께 설계하는 방법을 다룬다. |
| [08_alignment_safety_and_model_behavior](08_alignment_safety_and_model_behavior/README.md) | outlined | 모델 행동, 안전성, refusal/harmlessness/robustness를 체계적으로 읽는다. |

## 이 트랙에 포함되는 것

- language modeling objective, corpus/tokenizer 설계, pretraining mixture, domain adaptation
- instruction tuning, SFT, preference optimization, RLHF, reasoning-oriented RL
- RAG, evaluation harness, alignment/safety/model behavior framing

## 이 트랙에서 아직 다루지 않는 것

- torchrun, FSDP, tensor parallel 같은 분산 실행 기법은 `06_training_systems`에서 다룬다.
- 논문 재현·capstone·open-ended research 운영은 `07_frontier_labs`에서 다룬다.
- vision-text multimodal frontier는 `08_multimodal_bridge`, `09_multimodal`에서 다룬다.
