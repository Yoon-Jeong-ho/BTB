# BTB 대규모 커리큘럼 확장 디자인 스펙

## 1. 배경과 문제 정의

BTB는 현재 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서의 선형 학습 사다리를 중심으로, **작은 실험과 분석으로 딥러닝/NLP/멀티모달을 공부하는 한글 중심 저장소**로 정리되어 있다. 현재 구조는 기초에서 응용으로 올라가는 흐름이 분명하고, README / study guide / topology 테스트가 그 순서를 고정하고 있다는 점에서 강한 장점이 있다.

하지만 다음 문제가 남아 있다.

1. 딥러닝 코어의 폭이 아직 좁다. 현재 foundations는 텐서, activation/loss, gradient, normalization, runtime 중심이며, 사용자가 원하는 **퍼셉트론, MLP, CNN, 이미지 분류, RNN/LSTM/GRU, sequence modeling, GAN**까지는 아직 선형 코어 안에 충분히 반영되지 않았다.
2. NLP/LLM의 고급 축이 비어 있다. 현재 NLP는 applied task 중심이고, 사용자가 원하는 **pretraining, instruction tuning, RLHF, preference optimization, reasoning RL, verifier-based training, DP 계열, alignment/safety, RAG/eval**은 top-level 학습 구간으로 드러나지 않는다.
3. 학습 시스템/분산학습 축이 독립된 커리큘럼으로 존재하지 않는다. 사용자가 원하는 **torchrun, DDP, Accelerate, DeepSpeed, ZeRO/FSDP, tensor parallel, pipeline parallel, hybrid parallel, profiling/monitoring**은 향후 대형 모델 실습에 필수지만, 현재는 명시적 학습 계단이 없다.
4. 향후 저장소를 크게 키우려면, 단순히 unit를 늘리는 것이 아니라 **인덱스, 상태 모델, 문서 계약, 테스트, 에이전트 역할 문서**를 함께 확장해야 한다. 그렇지 않으면 구조는 커지는데 무엇이 완성되었고 무엇이 skeleton인지 구분이 흐려진다.
5. 사용자는 이 저장소를 “최고의 AI 이론/실습 저장소”로 키우고 싶어 하며, 이를 위해 **감독 에이전트, 이론 작성 에이전트, 조사 에이전트, 실험 실행 에이전트, 크리틱 에이전트**가 함께 작동하는 운영 모델까지 염두에 두고 있다.

이 설계의 목적은 BTB를 전면 재편하지 않고, **기존 00→05 사다리를 보존하면서 더 큰 AI 학습 프로그램으로 확장 가능한 구조**를 먼저 만드는 것이다.

## 2. 목표

### 핵심 목표
- 현재 `00→05` 코어 학습 사다리를 유지한다.
- foundations 내부에 딥러닝 심화 unit를 추가해, **퍼셉트론부터 시퀀스 데이터와 GAN까지** 선형 코어 안에서 이어지게 만든다.
- 고급 NLP/LLM과 학습 시스템/분산학습을 위한 **신규 top-level 트랙**을 추가한다.
- 이번 1차 확장에서는 **전체 커리큘럼 인덱스 + 트랙 문서 + unit 스켈레톤 계약 + 테스트 + 에이전트 역할 문서**를 먼저 깔아, 이후 확장을 쉽게 만든다.
- 각 unit가 `planned / outlined / runnable` 상태 중 어디에 있는지 명시하는 상태 모델을 도입한다.
- 기존 BTB의 강점인 **한글 중심 문서, 실험-분석 루프, 검증 테스트**를 새 트랙에도 동일하게 복제한다.

### 성공 기준
- 루트 README와 study guide만 읽어도 확장된 전체 사다리가 이해된다.
- 신규 top-level 트랙과 하위 unit가 번호/이름 규칙에 맞게 정렬된다.
- 각 unit의 상태가 문서와 테스트에서 일관되게 보인다.
- 사용자는 “지금 읽을 수 있는 것”과 “나중에 채워질 skeleton”을 구분할 수 있다.
- 에이전트 기반 확장을 위한 최소 역할 문서와 handoff 규칙이 생긴다.
- 이후 특정 unit를 runnable 상태로 높일 때, 이미 계약과 테스트 기반이 준비되어 있어 재작업이 최소화된다.

## 3. 비목표(Non-goals)

- 이번 단계에서 모든 신규 unit를 runnable하게 만들지 않는다.
- 현재 `00→05`를 전면 재번호 변경하지 않는다.
- 기존 코어 트랙을 삭제하거나 대거 이동하지 않는다.
- GPU 실험 자동 orchestration, job scheduler, idle GPU allocator 같은 운영 자동화를 이번 단계에서 완성하지 않는다.
- advanced LLM/system 주제를 모두 동일한 깊이로 곧바로 구현하지 않는다.
- 기존 dirty workspace나 unrelated artifact 문제를 함께 정리하는 cleanup 작업까지 확장하지 않는다.

## 4. 설계 원칙

1. **코어 사다리 보존**: 현재 `00→05`는 BTB의 정체성이므로 유지한다.
2. **확장 우선, 전면 재편 금지**: 이번 단계는 갈아엎기보다 구조적 확장에 집중한다.
3. **선형 가시성 유지**: 최상위에서는 여전히 번호 순서로 따라가게 하되, 내부에서 bridge/advanced/system 축을 분명히 드러낸다.
4. **상태 명시성**: skeleton unit를 완성된 unit처럼 보이게 하지 않는다.
5. **문서 계약 우선**: 인덱스만 추가하지 말고 README/THEORY/PREREQS/lesson metadata 계약을 함께 세운다.
6. **검증 가능성 우선**: 새 트랙도 topology, link, 상태, 계약 테스트를 갖는다.
7. **실험과 이론 분리**: theory writer와 experiment runner의 역할을 분리하고, critic/verifier가 merge gate를 잡는다.
8. **한글 우선, 기술 용어 병기**: 기존 BTB 문체 정책을 새 영역에도 유지한다.
9. **확장 가능한 numbering**: top-level 번호는 보존하되, 내부 unit numbering은 두 자리 이상으로 넓혀 future-proof하게 설계한다.

## 5. 선택지와 채택 결정

### 옵션 A — 채택
- 현재 `00→05`는 유지한다.
- `00_foundations` 내부를 확장해 딥러닝 심화 축을 흡수한다.
- `06_advanced_nlp_llm`, `07_training_systems`, `08_frontier_labs`를 신설한다.
- 신규 unit는 우선 skeleton/outlined 상태로 넓게 깐다.

### 옵션 B — 기각
- top-level 번호 체계를 전면 재편해 `01_deep_learning_core`, `02_vision`, `03_sequence_models`, `07_llm_posttraining`처럼 재배치한다.

### 왜 옵션 A를 채택하는가
- 현재 README / program map / study guide / topology tests가 이미 `00→05`를 강하게 고정하고 있다.
- 최근 커밋 흐름도 00~05 rollout 완료와 문서 정합성 강화에 맞춰져 있다.
- 옵션 B는 taxonomy는 예뻐질 수 있지만, 기존 사용자의 읽기 경로와 테스트/링크/문서 자산을 크게 깨뜨린다.
- 사용자는 선형 사다리를 유지하되 중간에 다리와 확장 구간을 붙이는 방식을 명시적으로 선호했다.

## 6. 목표 정보 구조(Top-level Architecture)

### 6.1 유지되는 top-level
```text
BTB/
├── 00_foundations/
├── 01_ml/
├── 02_nlp_bridge/
├── 03_nlp/
├── 04_multimodal_bridge/
├── 05_multimodal/
```

### 6.2 신규 top-level
```text
BTB/
├── 06_advanced_nlp_llm/
├── 07_training_systems/
└── 08_frontier_labs/
```

### 6.3 top-level 역할 요약
- `00_foundations/`: 딥러닝 코어 전체 축. 텐서부터 MLP/CNN/RNN/Transformer primer/GAN/runtime까지 포함
- `01_ml/`: 실험 discipline, metric 해석, baseline/interpretation의 안정적 첫 응용 구간
- `02_nlp_bridge/`: tokenization/embedding/attention/transformer anatomy로 NLP 입문 다리 제공
- `03_nlp/`: applied NLP task 학습
- `04_multimodal_bridge/`: text-only 표현에서 image-text alignment로 넘어가는 다리
- `05_multimodal/`: retrieval/captioning/VQA 등 멀티모달 applied task
- `06_advanced_nlp_llm/`: pretraining, SFT, preference optimization, RLHF, reasoning RL, RAG, alignment/eval
- `07_training_systems/`: distributed training, launcher/tooling, memory/offload, parallelism, monitoring
- `08_frontier_labs/`: 논문 재현, capstone, agentic experiment, benchmark construction

## 7. `00_foundations` 확장 설계

현재 foundations는 이미 코어 감각을 잘 잡고 있으므로, 방향을 바꾸지 않고 폭을 넓힌다.

### 7.1 목표 sequence
```text
00_foundations/
├── 01_tensor_shapes/
├── 02_perceptron_and_mlp/
├── 03_activation_and_loss/
├── 04_gradients_and_backpropagation/
├── 05_optimization_regularization_normalization/
├── 06_convolution_and_image_classification/
├── 07_sequence_models_rnn_lstm_gru/
├── 08_attention_and_transformer_primer/
├── 09_gpu_memory_runtime/
└── 10_generative_models_and_gan/
```

### 7.2 설계 의도
- `01_tensor_shapes`는 유지해 모든 후속 unit의 shape 감각 기반을 제공한다.
- `02_perceptron_and_mlp`를 추가해 “딥러닝이 무엇인가”를 가장 작은 supervised 모델로 명시한다.
- activation/loss, gradients/backprop, optimization/regularization을 분리해 내부 숫자 흐름을 단계별로 본다.
- CNN/image classification을 foundations 내부에 둬서 비전도 코어 딥러닝의 일부로 본다.
- RNN/LSTM/GRU/sequence data를 foundations에 두어 NLP bridge 이전에 sequence intuition을 심는다.
- Transformer primer는 NLP bridge의 attention unit와 중복되지 않도록, foundations에서는 “딥러닝 코어 관점의 sequence mixer”로 다루고, NLP bridge에서는 “문장을 모델이 읽는 과정”에 더 집중한다.
- GAN은 “완전한 최신 generative modeling 백과사전”이 아니라, generator/discriminator/adversarial objective를 이해하는 코어 단위로 둔다.

## 8. `06_advanced_nlp_llm` 설계

### 8.1 목표 sequence
```text
06_advanced_nlp_llm/
├── 01_language_modeling_and_pretraining_objectives/
├── 02_corpus_tokenizer_and_data_mixture/
├── 03_domain_adaptive_pretraining/
├── 04_instruction_tuning_and_sft/
├── 05_preference_optimization_dpo_orpo_kto/
├── 06_rlhf_and_reasoning_rl/
├── 07_retrieval_augmented_generation_and_eval/
└── 08_alignment_safety_and_model_behavior/
```

### 8.2 scope mapping
- `01`: causal LM / masked LM / span corruption / objective design
- `02`: tokenizer trade-offs, corpus quality, data mixture, dedup, contamination awareness
- `03`: domain continued pretraining, specialty corpora, transfer trade-offs
- `04`: supervised fine-tuning, instruction data formatting, response style control
- `05`: DPO / ORPO / KTO / pairwise preference optimization 계열
- `06`: RLHF, reward modeling, reasoning RL, verifier-guided post-training, RLCR/RLAIF 유사 계열을 이 unit에서 정리
- `07`: RAG pipeline, retriever-reader coupling, retrieval eval, grounding metrics
- `08`: alignment, safety, refusal/over-refusal, behavior eval, lightweight policy tests

### 8.3 DP / RLCR 처리 원칙
사용자가 언급한 `DP`, `RLCR`은 구체적 의미가 문맥에 따라 달라질 수 있다. 1차 skeleton에서는 다음처럼 다룬다.
- `DP`는 문맥상 **Differential Privacy, Direct Preference 계열 약칭, 또는 다른 post-training variant**를 의미할 수 있으므로, skeleton 단계에서는 `06_advanced_nlp_llm`의 post-training note로만 걸고 implementation plan에서 정확한 expansion을 고정한다.
- `RLCR`도 특정 community-specific 약어일 가능성이 높으므로, skeleton 단계에서는 **reasoning / reward / control 계열 RL variant bucket**으로만 다루고 runnable unit 전환 시 대표 논문과 정확한 명칭을 확정한다.

즉 skeleton 단계에서는 **주제 bucket은 포함하되, 모호한 약어는 implementation 전에 명시적으로 해소**한다.

## 9. `07_training_systems` 설계

### 9.1 목표 sequence
```text
07_training_systems/
├── 01_torchrun_and_ddp_basics/
├── 02_accelerate_workflows/
├── 03_deepspeed_zero/
├── 04_fsdp_checkpointing_and_offload/
├── 05_tensor_parallelism/
├── 06_pipeline_parallelism/
├── 07_data_parallel_grad_accumulation/
├── 08_hybrid_parallel_topologies/
└── 09_profiling_monitoring_and_failure_recovery/
```

### 9.2 설계 의도
- 사용자가 열거한 학습 기법들을 한 트랙에 정리해 “모델 학습 자체를 이해하는 커리큘럼”으로 만든다.
- 앞쪽은 launcher/tooling (`torchrun`, `Accelerate`, DDP) 중심으로 가고, 뒤쪽은 memory partitioning/parallel topology (`DeepSpeed`, `FSDP`, tensor/pipeline/hybrid parallel`)로 간다.
- 마지막 unit는 profiler, checkpoint, resume, OOM, throughput 분석, failure recovery 등 운영 현실을 다룬다.

## 10. `08_frontier_labs` 설계

### 10.1 목표 sequence
```text
08_frontier_labs/
├── 01_paper_reproduction_playground/
├── 02_capstone_model_building/
├── 03_agentic_training_and_eval_loops/
├── 04_benchmark_and_dataset_construction/
└── 05_open_ended_research_tracks/
```

### 10.2 역할
- 이 트랙은 foundations나 advanced track처럼 완전한 선행 필수 구간이 아니라, 상위 학습이 어느 정도 쌓인 뒤 들어가는 연구형 샌드박스다.
- 논문 재현, 캡스톤, 에이전트형 training/eval loop, benchmark 제작, open-ended exploration을 담는다.
- “최고의 AI 이론/실습 저장소”라는 비전은 이 트랙에서 가장 직접적으로 드러난다.

## 11. 승인된 1차 확장 범위

이번 승인된 범위는 **인덱스 + 문서 + unit 계약 + 테스트 + 기본 에이전트 역할 문서**다.

### 11.1 반드시 포함할 것
- 루트 `README.md`의 전체 사다리 확장
- `docs/00_program_map.md`의 전체 프로그램 설명 확장
- `docs/02_study_guide.md`의 확장된 읽기 순서 추가
- 신규 top-level (`06`, `07`, `08`) README 생성
- `00_foundations` 확장 unit skeleton 반영
- 신규 unit 디렉토리 skeleton 생성
- 상태 모델 문서화
- topology / track docs / status tests 추가
- 에이전트 역할 문서 추가

### 11.2 일부만 추가하고 나중에 미룰 것
- runnable labs
- 실제 dataset/model selection 문서의 상세 reference
- GPU job orchestration
- reward model / RLHF / distributed runtime의 실제 heavy experiment

## 12. unit 상태 모델

### 12.1 상태 정의
- `planned`: 인덱스와 README 수준의 자리만 확보된 상태
- `outlined`: `README.md`, `THEORY.md`, `PREREQS.md`, `lesson.yaml`까지 갖춘 상태
- `runnable`: `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`, artifacts contract, tests까지 갖춘 상태

### 12.2 왜 필요한가
BTB가 커질수록 “다 있는 것처럼 보이는데 사실은 인덱스만 있는 unit”가 생길 수 있다. 상태 모델은 다음을 가능하게 한다.
- 학습자가 지금 실제로 읽고 실행할 수 있는 범위를 빠르게 안다.
- maintainer가 skeleton debt를 추적할 수 있다.
- 테스트가 unit별 최소 contract를 상태에 따라 다르게 검증할 수 있다.

### 12.3 상태 표현 방식
- 각 unit `README.md` 상단에 status badge 또는 status 문장을 둔다.
- `lesson.yaml` 또는 별도 metadata field에 `status: planned|outlined|runnable`를 명시한다.
- 상위 track README에 unit 상태 표를 노출한다.

## 13. 상태별 계약

### 13.1 planned
필수:
- `README.md`
- top-level / track index 연결
- status 표기
- “왜 이 unit가 필요한가 / 앞으로 무엇이 들어갈 것인가 / 선행 개념은 무엇인가” 수준의 문서

### 13.2 outlined
필수:
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `artifacts/.gitkeep`
- 상태 표기 및 required outputs 설계

### 13.3 runnable
필수:
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.py`
- `analysis.md`
- `reflection.md`
- `artifacts/.gitkeep`
- state-aware tests

## 14. 문서 계약

### 14.1 unit README가 반드시 말해야 할 것
- 왜 이 단위를 배우는가
- 이번 단위에서 남길 것
- 실습 흐름
- 실행 결과 예시(또는 skeleton인 경우 future expected outputs)
- 다음 단위와의 연결
- 현재 상태(`planned`, `outlined`, `runnable`)

### 14.2 THEORY 계약
- 핵심 개념 정의
- 흔한 오해
- 관찰해야 할 숫자/도표
- 이후 unit로 이어지는 질문

### 14.3 PREREQS 계약
- 선행 개념 체크리스트
- 모르면 다시 볼 이전 unit 링크

### 14.4 lesson.yaml 계약
최소 필드:
- `objective`
- `status`
- `prereqs`
- `key_terms`
- `required_outputs`
- `analysis_questions`
- `datasets_or_inputs`
- `models_or_tools`

## 15. 검증과 테스트 확장

### 15.1 추가해야 할 테스트 종류
- 확장된 top-level ladder order 테스트
- 신규 track README 존재 테스트
- study guide에 신규 track가 노출되는지 테스트
- 상태 모델 일관성 테스트
- planned/outlined/runnable별 contract 테스트
- agent role docs 존재 테스트

### 15.2 link checker 정책
- 새 spec/track/unit 문서는 link checker 대상에 포함되므로, 상대경로/절대경로 정책을 일관되게 맞춘다.
- 현재 repo에는 기존 문서 link checker failure가 있으므로, 확장 작업 중에는 새 문서가 기존 failure를 악화시키지 않도록 한다.
- 확장 구현 단계에서 기존 broken link 정리 여부는 별도 change set으로 분리하는 것이 바람직하다.

## 16. 에이전트 조직 설계

사용자가 원하는 운영 모델을 1차에서는 “실행 자동화”가 아니라 **역할 문서 + handoff contract** 수준으로 설계한다.

### 16.1 역할
- **Program Director / Supervisor**
  - 전체 로드맵, 우선순위, 병렬 lane 조정, 최종 gate 관리
- **Curriculum Architect**
  - 트랙 구조, numbering, prerequisite graph, status progression 설계
- **Theory Writer**
  - README / THEORY / PREREQS / reflection skeleton 작성
- **Researcher / Data Scout**
  - 논문, 데이터셋, benchmark, reference implementation 조사
- **Experiment Runner**
  - 유휴 GPU/CPU 자원에서 실험을 실행하고 artifacts를 남김
- **Critic / Verifier**
  - 문서 품질, 링크, contract, 실험 산출물, claim/evidence 일치 여부 검증

### 16.2 1차 산출물
- 각 역할별 책임 범위 문서
- handoff 입력/출력 contract
- “theory-first lane / experiment lane / critic gate” 흐름도

### 16.3 아직 미루는 것
- 자동 idle GPU 탐지
- job queue allocator
- persistent agent memory for experiment scheduling
- automated merge gate bots

## 17. 권장 rollout 순서

1. top-level architecture와 index 문서 확장
2. `00_foundations` 확장 unit skeleton 추가
3. `06_advanced_nlp_llm` skeleton 추가
4. `07_training_systems` skeleton 추가
5. `08_frontier_labs` skeleton 추가
6. 상태 모델 문서와 테스트 추가
7. 에이전트 역할 문서 추가
8. 이후 개별 unit를 `planned -> outlined -> runnable`로 승격

## 18. 리스크와 완화책

### 리스크 1 — foundations 과대비대화
- **문제**: `00_foundations`가 너무 커져서 entry barrier가 높아질 수 있다.
- **완화**: study guide에 `core-first route`와 `full-deep-learning route`를 분리해 제시한다.

### 리스크 2 — skeleton 과다 생성
- **문제**: unit 수는 늘었지만 실제 runnable 비율이 낮아져 체감 품질이 떨어질 수 있다.
- **완화**: 상태 모델을 강하게 노출하고, track별 runnable coverage 표를 둔다.

### 리스크 3 — 고급 약어/기법의 의미 모호성
- **문제**: DP, RLCR 같은 약어가 문맥마다 다를 수 있다.
- **완화**: skeleton 단계에서는 bucket만 잡고, runnable 전환 시 acronym clarification section을 필수화한다.

### 리스크 4 — 분산학습 트랙의 실행 비용
- **문제**: 07 트랙은 실제 재현 비용이 높다.
- **완화**: 1차는 docs/contract 위주로 두고, runnable 단계에서는 CPU-safe toy + small-scale DDP demos부터 시작한다.

### 리스크 5 — 에이전트 운영 과설계
- **문제**: 역할 문서는 늘어나는데 실제 운영이 없으면 문서만 비대해질 수 있다.
- **완화**: 1차는 역할과 handoff contract만 작성하고, 실제 automation은 Frontier Labs 또는 후속 계획으로 분리한다.

## 19. 구현 전 확인할 후속 질문

이 스펙은 1차 구조 확장을 위한 것이므로, 실제 implementation plan 단계에서 아래를 구체화해야 한다.

1. `00_foundations` 신규 unit 중 어떤 순서로 runnable 승격을 시작할 것인가?
2. `06_advanced_nlp_llm`에서 acronym ambiguity(`DP`, `RLCR`)를 어떤 공식 용어로 고정할 것인가?
3. agent role docs는 `.codex/` 아래 prompt/skill 형태로 둘지, `docs/agents/` 설명 문서로 둘지?
4. status metadata를 `lesson.yaml`에 둘지 별도 manifest에 둘지?
5. 확장된 study guide를 “전체 1-pass”와 “분기형 압축 루트” 두 버전으로 모두 제공할지?

## 20. 최종 권고

이번 단계는 **코어 사다리를 깨지 않고 전체 지도를 크게 넓히는 작업**으로 정의해야 한다. 가장 좋은 방향은 다음과 같다.

1. `00→05`는 코어로 유지한다.
2. `00_foundations`를 딥러닝 전반까지 확장한다.
3. `06_advanced_nlp_llm`, `07_training_systems`, `08_frontier_labs`를 추가한다.
4. 인덱스만 늘리지 말고, 상태 모델 / 문서 계약 / 테스트 / 에이전트 역할 문서까지 함께 깐다.
5. 그 위에서 개별 unit를 점진적으로 runnable 상태로 승격한다.

이 방향이 현재 BTB의 자산을 가장 잘 살리면서, 이후 대형 확장과 병렬 제작을 가장 쉽게 만든다.
