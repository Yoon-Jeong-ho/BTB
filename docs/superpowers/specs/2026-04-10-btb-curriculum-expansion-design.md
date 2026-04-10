# BTB 대규모 커리큘럼 확장 디자인 스펙

## 1. 배경과 문제 정의

BTB는 현재 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서의 선형 학습 사다리를 중심으로, **작은 실험과 분석으로 딥러닝/NLP/멀티모달을 공부하는 한글 중심 저장소**로 정리되어 있다. 현재 구조는 기초에서 응용으로 올라가는 흐름이 분명하고, README / study guide / topology 테스트가 그 순서를 고정하고 있다는 점에서 강한 장점이 있다.

하지만 사용자의 확장 요구를 기준으로 보면, 이전 설계 초안에는 구조적 문제가 있었다.

1. 딥러닝 내용을 모두 `00_foundations` 안으로 밀어 넣으면 `01_ml`의 위치가 애매해진다. foundations가 지나치게 비대해지고, 딥러닝 본체와 공통 기초의 경계가 흐려진다.
2. 사용자가 원하는 딥러닝 범위는 단순 foundations 보강이 아니라, **퍼셉트론, MLP, CNN, 이미지 분류, sequence model, attention/transformer, GAN/VAE 계열**까지 포함하는 독립된 학습 트랙에 가깝다.
3. 고급 NLP/LLM, 학습 시스템/분산학습, frontier lab를 NLP 뒤에 배치하고, 멀티모달은 더 뒤로 보내고 싶다는 사용자 의도가 분명해졌다. 즉 이전의 “멀티모달 후 advanced track” 설계는 의도와 어긋난다.
4. 향후 저장소를 크게 키우려면, 단순히 디렉토리만 늘리는 것이 아니라 **인덱스, 상태 모델, 문서 계약, 테스트, 에이전트 역할 문서**를 함께 확장해야 한다. 그렇지 않으면 무엇이 skeleton이고 무엇이 runnable인지 구분이 흐려진다.
5. 사용자는 이 저장소를 “최고의 AI 이론/실습 저장소”로 키우고 싶어 하며, 이를 위해 **감독 에이전트, 이론 작성 에이전트, 조사 에이전트, 실험 실행 에이전트, 크리틱/검증 에이전트**가 함께 작동하는 운영 모델까지 염두에 두고 있다.

이 설계의 목적은 BTB를 전면 재편하지 않고, **선형 사다리는 유지하되 `02_deep_learning`을 신설하고, NLP 이후 고급 구간을 재배치하고, 멀티모달을 후반부로 이동시키는 새로운 확장 구조**를 먼저 만드는 것이다.

## 2. 목표

### 핵심 목표
- 선형적인 top-level 사다리는 유지한다.
- `00_foundations`는 **공통 기초**로 얇고 명확하게 유지한다.
- `01_ml`는 **실험 discipline과 baseline 해석 트랙**으로 유지한다.
- `02_deep_learning`를 신설해, 사용자가 원하는 본격 딥러닝 아키텍처 학습을 담당하게 한다.
- NLP 구간을 `03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`의 3단 구조로 재편한다.
- `06_training_systems`를 신설해 distributed training과 large-model training technique를 독립 트랙으로 분리한다.
- frontier labs를 NLP 이후에 두고, multimodal은 더 뒤인 `08_multimodal_bridge -> 09_multimodal`로 이동시킨다.
- 이번 1차 확장에서는 **전체 커리큘럼 인덱스 + 트랙 문서 + unit 스켈레톤 계약 + 테스트 + 에이전트 역할 문서**를 먼저 깐다.
- 각 unit가 `planned / outlined / runnable` 상태 중 어디에 있는지 명시하는 상태 모델을 도입한다.

### 성공 기준
- 루트 README와 study guide만 읽어도 확장된 전체 사다리가 이해된다.
- 신규 top-level 트랙과 하위 unit가 번호/이름 규칙에 맞게 정렬된다.
- `00_foundations`, `01_ml`, `02_deep_learning`의 역할 경계가 명확하다.
- NLP 이후 고급부(`05`, `06`, `07`)와 multimodal 후반부(`08`, `09`)의 순서가 자연스럽다.
- 각 unit의 상태가 문서와 테스트에서 일관되게 보인다.
- 사용자는 “지금 읽을 수 있는 것”과 “나중에 채워질 skeleton”을 구분할 수 있다.
- 이후 특정 unit를 runnable 상태로 높일 때, 이미 계약과 테스트 기반이 준비되어 있어 재작업이 최소화된다.

## 3. 비목표(Non-goals)

- 이번 단계에서 모든 신규 unit를 runnable하게 만들지 않는다.
- 기존 코어 내용을 대규모 삭제/이동하지 않는다.
- GPU 실험 자동 orchestration, idle GPU 스케줄러, job allocator를 이번 단계에서 완성하지 않는다.
- advanced LLM/system topic을 곧바로 heavy experiment까지 완성하지 않는다.
- 기존 dirty workspace나 unrelated artifact 문제를 이 설계 변경과 한 change set에서 같이 정리하지 않는다.

## 4. 설계 원칙

1. **선형 가시성 유지**: top-level은 계속 번호 순서로 따라갈 수 있어야 한다.
2. **공통 기초와 본격 딥러닝 분리**: `00_foundations`와 `02_deep_learning`의 역할을 섞지 않는다.
3. **실험 discipline 독립 유지**: `01_ml`는 DL architecture survey가 아니라 실험 읽는 법을 먼저 익히는 곳으로 둔다.
4. **NLP 3단 구조**: bridge, applied, advanced LLM을 분리한다.
5. **시스템 독립 트랙화**: distributed/system topic을 LLM이나 multimodal 부록으로 두지 않는다.
6. **멀티모달 후반부 배치**: multimodal은 후반 응용 확장으로 보낸다.
7. **상태 명시성**: skeleton unit를 완성된 unit처럼 보이게 하지 않는다.
8. **문서 계약 우선**: 인덱스만 추가하지 말고 README/THEORY/PREREQS/lesson metadata 계약을 함께 세운다.
9. **검증 가능성 우선**: 새 트랙도 topology, link, 상태, 계약 테스트를 갖는다.
10. **실험과 이론 분리**: theory writer와 experiment runner의 역할을 분리하고, critic/verifier가 merge gate를 잡는다.

## 5. 선택지와 채택 결정

### 기존 초안 — 기각
- `00_foundations` 내부를 크게 확장한다.
- multimodal을 `05` 근처에 유지하고, advanced NLP/system/frontier는 그 뒤에 붙인다.

### 수정안 — 채택
- `00_foundations`는 현재처럼 compact foundation track으로 유지한다.
- `01_ml`는 실험 discipline track으로 유지한다.
- `02_deep_learning`를 신설한다.
- 기존 `02_nlp_bridge`는 `03_nlp_bridge`로, 기존 `03_nlp`는 `04_nlp`로 민다.
- 그 뒤에 `05_advanced_nlp_llm`, `06_training_systems`, `07_frontier_labs`를 둔다.
- 멀티모달은 `08_multimodal_bridge`, `09_multimodal`로 후반부로 이동시킨다.

### 왜 수정안을 채택하는가
- 사용자가 직접 “00에 딥러닝을 다 넣으면 01 ML이 애매하다”고 문제를 제기했다.
- 서브에이전트 분석도 `00_foundations`와 `02_deep_learning`의 역할을 분리하는 쪽이 더 깔끔하다고 지지했다.
- 이 구조는 현재 코어 자산을 최대한 살리면서도, 딥러닝 본체를 독립 트랙으로 크게 넓힐 수 있다.
- NLP 이후에 고급 NLP/LLM, 시스템, frontier를 연달아 배치하고 multimodal을 뒤로 보내려는 사용자 의도를 자연스럽게 반영한다.

## 6. 채택된 최상위 정보 구조

### 6.1 최상위 순서
```text
BTB/
├── 00_foundations/
├── 01_ml/
├── 02_deep_learning/
├── 03_nlp_bridge/
├── 04_nlp/
├── 05_advanced_nlp_llm/
├── 06_training_systems/
├── 07_frontier_labs/
├── 08_multimodal_bridge/
└── 09_multimodal/
```

### 6.2 기존 대비 인덱스 변경
- 기존 `02_nlp_bridge` → `03_nlp_bridge`
- 기존 `03_nlp` → `04_nlp`
- 기존 `04_multimodal_bridge` → `08_multimodal_bridge`
- 기존 `05_multimodal` → `09_multimodal`
- 신규 `02_deep_learning`, `05_advanced_nlp_llm`, `06_training_systems`, `07_frontier_labs` 추가

## 7. 트랙별 역할 경계

### 7.1 `00_foundations`
**역할:** 모든 후속 트랙이 공유하는 가장 얇은 공통 바닥

포함:
- tensor shape / broadcasting / matmul
- activation / logits / loss 기초
- gradient / backprop / autograd 기초
- optimization / regularization / normalization 기초
- dtype / GPU memory / runtime / numerical intuition

제외:
- CNN / image classification 전체 실습
- RNN / LSTM / GRU / sequence modeling 본체
- transformer 본격 구조
- GAN / VAE
- task-level NLP / multimodal
- distributed systems / RLHF / pretraining

즉 `00_foundations`는 **숫자 흐름과 실행 감각의 최소 바닥**이다.

### 7.2 `01_ml`
**역할:** 실험 discipline, baseline, metric 해석, artifact 습관을 익히는 트랙

포함:
- tabular classification / regression
- validation / calibration / residual / interpretation
- failure analysis
- experiment artifact discipline

제외:
- representation learning 본체
- neural architecture survey
- distributed training systems
- RLHF / post-training

즉 `01_ml`는 **모델이 왜 맞고 틀리는지 읽는 법**을 먼저 익히는 곳이다.

### 7.3 `02_deep_learning`
**역할:** 본격적인 딥러닝 모델 패밀리 학습 트랙

포함:
- perceptron / MLP
- CNN / image classification
- RNN / LSTM / GRU / sequence modeling
- attention / transformer core
- autoencoder / representation learning
- VAE / GAN / diffusion preview
- training recipes / debugging / transfer learning basics

제외:
- tokenization / LM objective / NER / MRC 같은 NLP task specifics
- distributed training systems
- instruction tuning / RLHF / DPO 계열

즉 `02_deep_learning`은 **퍼셉트론부터 sequence data, transformer, GAN까지 배우는 딥러닝 본체**다.

### 7.4 `03_nlp_bridge`
**역할:** DL에서 NLP로 넘어가는 연결 다리

포함:
- tokenization
- embeddings
- positional encoding / text tensorization
- masking
- self-attention을 NLP 입력 관점에서 설명
- transformer block의 text-specific 입구 감각

제외:
- 본격 NLP task 실습
- pretraining / SFT / RLHF
- serving / distributed systems

즉 `03_nlp_bridge`는 **문장이 숫자로 들어가고 섞이는 방식**을 다룬다.

### 7.5 `04_nlp`
**역할:** applied NLP core

포함:
- text classification
- sequence labeling / NER
- MRC / QA
- 필요 시 semantic search 기초 확장

제외:
- pretraining
- instruction tuning
- RLHF / DPO / ORPO / PPO 계열
- distributed LLM training

즉 `04_nlp`는 **표준 NLP task를 돌리고 해석하는 트랙**이다.

### 7.6 `05_advanced_nlp_llm`
**역할:** pretraining 이후의 고급 NLP/LLM 트랙

포함:
- language modeling / pretraining objective
- corpus / tokenizer / data mixture
- domain adaptive pretraining
- instruction tuning / SFT
- preference optimization (`DPO`, `ORPO`, `KTO` 등)
- RLHF / reasoning RL / verifier-based post-training
- RAG / retrieval eval / alignment eval / safety behavior

주의:
- 사용자가 언급한 `DP`, `RLCR`은 약어 의미가 모호하므로 skeleton 단계에서는 bucket만 확보하고, runnable 전환 시 정확한 expansion을 고정한다.

### 7.7 `06_training_systems`
**역할:** large-model training system과 distributed training을 배우는 독립 트랙

포함:
- `torchrun`, DDP
- Hugging Face Accelerate
- DeepSpeed / ZeRO
- FSDP / checkpointing / offload
- tensor parallel / pipeline parallel / data parallel
- hybrid parallel topology
- profiling / monitoring / failure recovery

원칙:
- 1차 확장에서는 theory + small demo + contract 중심으로 두고, heavy multi-GPU run은 후속 단계로 넘긴다.

### 7.8 `07_frontier_labs`
**역할:** 고급 NLP/시스템 이후의 연구형 sandbox

포함:
- paper reproduction
- capstone model building
- agentic training/eval loops
- benchmark / dataset construction
- open-ended research tracks

이 트랙은 “최종 terminal capstone”이라기보다, **후반부 고급 구간에서 병렬적으로 열 수 있는 실험 허브**로 본다.

### 7.9 `08_multimodal_bridge`
**역할:** multimodal 직전 연결 다리

포함:
- contrastive alignment
- vision-text representation coupling
- retrieval vs generation transition
- grounding / cross-attention intuition

### 7.10 `09_multimodal`
**역할:** multimodal applied track

포함:
- image-text retrieval
- captioning
- VQA
- grounding / reasoning / failure analysis

이렇게 multimodal을 뒤로 보내면, 텍스트/LLM/시스템을 충분히 본 뒤에 **멀티모달을 후반 응용 확장**으로 받아들일 수 있다.

## 8. `02_deep_learning` 세부 골격

### 8.1 권장 unit 목록
```text
02_deep_learning/
├── 01_perceptron_and_mlp/
├── 02_cnn_and_image_classification/
├── 03_sequence_models_rnn_lstm_gru/
├── 04_attention_and_transformers/
├── 05_autoencoders_and_representation_learning/
├── 06_generative_models_vae_gan/
└── 07_training_recipes_and_debugging/
```

### 8.2 설계 의도
- `01`: 퍼셉트론, single neuron, MLP, hidden layer intuition
- `02`: convolution, pooling, residual intuition, image classification baseline
- `03`: RNN/LSTM/GRU, sequence modeling, teacher forcing, vanishing/exploding intuition
- `04`: attention / transformer를 general DL 관점에서 설명
- `05`: latent representation, autoencoder, embedding space intuition
- `06`: VAE/GAN과 adversarial/generative objective 감각
- `07`: scheduler, augmentation, class imbalance, debug recipe, overfit diagnosis

### 8.3 `00_foundations`와의 경계
- `00_foundations`는 activation/loss/backprop/runtime의 바닥
- `02_deep_learning`은 architecture family와 training behavior의 본체

즉 `00`이 “왜 숫자가 이렇게 흐르는가”라면, `02`는 “이 숫자 흐름이 어떤 모델 구조로 구현되는가”다.

## 9. `05_advanced_nlp_llm` 세부 골격

```text
05_advanced_nlp_llm/
├── 01_language_modeling_and_pretraining_objectives/
├── 02_corpus_tokenizer_and_data_mixture/
├── 03_domain_adaptive_pretraining/
├── 04_instruction_tuning_and_sft/
├── 05_preference_optimization_dpo_orpo_kto/
├── 06_rlhf_and_reasoning_rl/
├── 07_retrieval_augmented_generation_and_eval/
└── 08_alignment_safety_and_model_behavior/
```

핵심 원칙:
- `05`는 “LLM을 어떻게 만들고 정렬하는가”를 다룬다.
- `06`은 “그 큰 모델을 어떻게 효율적으로 학습하는가”를 다룬다.
- 즉 모델링과 시스템을 섞지 않는다.

## 10. `06_training_systems` 세부 골격

```text
06_training_systems/
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

설계 원칙:
- 앞쪽은 launcher/tooling (`torchrun`, `Accelerate`, DDP)
- 중간은 memory partitioning (`DeepSpeed`, `ZeRO`, `FSDP`)
- 뒤쪽은 topology (`tensor`, `pipeline`, `hybrid`)
- 마지막은 profiler / checkpoint / OOM / failure recovery

## 11. `07_frontier_labs` 세부 골격

```text
07_frontier_labs/
├── 01_paper_reproduction_playground/
├── 02_capstone_model_building/
├── 03_agentic_training_and_eval_loops/
├── 04_benchmark_and_dataset_construction/
└── 05_open_ended_research_tracks/
```

역할:
- 논문 재현
- capstone
- agentic pipeline 실험
- benchmark/dataset design
- open-ended research branch

이 트랙은 multimodal보다 앞에 오지만, multimodal을 대체하지 않는다. **NLP/LLM/system 기반의 연구형 sandbox**로 해석한다.

## 12. 승인된 1차 확장 범위

이번 승인된 범위는 **인덱스 + 문서 + unit 계약 + 테스트 + 기본 에이전트 역할 문서**다.

### 12.1 반드시 포함할 것
- 루트 `README.md`의 전체 사다리 확장
- `docs/00_program_map.md`의 전체 프로그램 설명 확장
- `docs/02_study_guide.md`의 확장된 읽기 순서 추가
- 신규 top-level (`02`, `05`, `06`, `07`) README 생성 및 renumbered track 인덱스 반영
- `02_deep_learning` skeleton 추가
- `03_nlp_bridge`, `04_nlp`, `08_multimodal_bridge`, `09_multimodal`의 renumbering plan 반영
- 상태 모델 문서화
- topology / track docs / status tests 추가
- 에이전트 역할 문서 추가

### 12.2 일부만 추가하고 나중에 미룰 것
- runnable labs
- 실제 dataset/model selection 상세 reference
- idle GPU orchestration
- heavy distributed runtime 실험
- RLHF / reward model / multimodal large-scale training의 full implementation

## 13. unit 상태 모델

### 13.1 상태 정의
- `planned`: 인덱스와 README 수준의 자리만 확보된 상태
- `outlined`: `README.md`, `THEORY.md`, `PREREQS.md`, `lesson.yaml`까지 갖춘 상태
- `runnable`: `scratch_lab.py`, `framework_lab.py`, `analysis.py`, `analysis.md`, `reflection.md`, artifacts contract, tests까지 갖춘 상태

### 13.2 왜 필요한가
BTB가 커질수록 “다 있는 것처럼 보이는데 사실은 인덱스만 있는 unit”가 생길 수 있다. 상태 모델은 다음을 가능하게 한다.
- 학습자가 지금 실제로 읽고 실행할 수 있는 범위를 빠르게 안다.
- maintainer가 skeleton debt를 추적할 수 있다.
- 테스트가 unit별 최소 contract를 상태에 따라 다르게 검증할 수 있다.

### 13.3 상태 표현 방식
- 각 unit `README.md` 상단에 status badge 또는 status 문장을 둔다.
- `lesson.yaml` 또는 별도 metadata field에 `status: planned|outlined|runnable`를 명시한다.
- 상위 track README에 unit 상태 표를 노출한다.

## 14. 상태별 계약

### 14.1 planned
필수:
- `README.md`
- top-level / track index 연결
- status 표기
- “왜 이 unit가 필요한가 / 앞으로 무엇이 들어갈 것인가 / 선행 개념은 무엇인가” 수준의 문서

### 14.2 outlined
필수:
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `artifacts/.gitkeep`
- 상태 표기 및 required outputs 설계

### 14.3 runnable
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

## 15. 문서 계약

### 15.1 unit README가 반드시 말해야 할 것
- 왜 이 단위를 배우는가
- 이번 단위에서 남길 것
- 실습 흐름
- 실행 결과 예시(또는 skeleton인 경우 future expected outputs)
- 다음 단위와의 연결
- 현재 상태(`planned`, `outlined`, `runnable`)

### 15.2 THEORY 계약
- 핵심 개념 정의
- 흔한 오해
- 관찰해야 할 숫자/도표
- 이후 unit로 이어지는 질문

### 15.3 PREREQS 계약
- 선행 개념 체크리스트
- 모르면 다시 볼 이전 unit 링크

### 15.4 lesson.yaml 계약
최소 필드:
- `objective`
- `status`
- `prereqs`
- `key_terms`
- `required_outputs`
- `analysis_questions`
- `datasets_or_inputs`
- `models_or_tools`

## 16. 검증과 테스트 확장

### 16.1 추가해야 할 테스트 종류
- 확장된 top-level ladder order 테스트
- 신규/renumbered track README 존재 테스트
- study guide에 신규 사다리가 노출되는지 테스트
- 상태 모델 일관성 테스트
- planned/outlined/runnable별 contract 테스트
- agent role docs 존재 테스트

### 16.2 link checker 정책
- 새 spec/track/unit 문서는 link checker 대상에 포함되므로, 상대경로/절대경로 정책을 일관되게 맞춘다.
- 현재 repo에는 기존 문서 link checker failure가 있으므로, 확장 작업 중에는 새 문서가 기존 failure를 악화시키지 않도록 한다.
- renumbering을 실제로 적용하는 implementation 단계에서는 redirect note 또는 migration table이 필요하다.

## 17. 에이전트 조직 설계

사용자가 원하는 운영 모델을 1차에서는 “실행 자동화”가 아니라 **역할 문서 + handoff contract** 수준으로 설계한다.

### 17.1 역할
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

### 17.2 1차 산출물
- 각 역할별 책임 범위 문서
- handoff 입력/출력 contract
- theory-first lane / experiment lane / critic gate 흐름도

### 17.3 아직 미루는 것
- 자동 idle GPU 탐지
- job queue allocator
- persistent agent memory for experiment scheduling
- automated merge gate bot

## 18. 권장 rollout 순서

1. top-level architecture와 index 문서 재배치
2. `02_deep_learning` skeleton 추가
3. `05_advanced_nlp_llm` skeleton 추가
4. `06_training_systems` skeleton 추가
5. `07_frontier_labs` skeleton 추가
6. multimodal renumbering plan(`08`, `09`) 문서 반영
7. 상태 모델 문서와 테스트 추가
8. 에이전트 역할 문서 추가
9. 이후 개별 unit를 `planned -> outlined -> runnable`로 승격

## 19. 리스크와 완화책

### 리스크 1 — `00_foundations`와 `02_deep_learning`의 역할 충돌
- **문제**: activation/gradient/runtime과 MLP/CNN/RNN/Transformer/GAN이 섞이면 경계가 다시 흐려질 수 있다.
- **완화**: 각 track README 첫 단락에 포함/제외 범위를 명시하고, study guide에서 00과 02의 학습 목적을 분리한다.

### 리스크 2 — NLP 이후 고급부 과밀화
- **문제**: advanced NLP/LLM, training systems, frontier labs를 한꺼번에 붙이면 난이도 급상승과 의미 중복이 생길 수 있다.
- **완화**: `05=모델링`, `06=시스템`, `07=연구 sandbox`로 역할을 강하게 분리한다.

### 리스크 3 — multimodal이 뒤로 밀리며 연결감 약화
- **문제**: 기존 repo의 multimodal 흐름이 자연스러웠는데, 뒤로 보내면 동기 부여가 약해질 수 있다.
- **완화**: `08_multimodal_bridge`를 유지하고, root/program map에서 multimodal이 왜 후반부에 오는지 명시한다.

### 리스크 4 — skeleton 과다 생성
- **문제**: unit 수는 늘었지만 runnable 비율이 낮아져 체감 품질이 떨어질 수 있다.
- **완화**: 상태 모델을 강하게 노출하고 track별 runnable coverage 표를 둔다.

### 리스크 5 — 분산학습 트랙의 실행 비용
- **문제**: `06_training_systems`는 실제 재현 비용이 높다.
- **완화**: 1차는 docs/contract 위주로 두고, runnable 단계에서는 CPU-safe toy + small-scale DDP demo부터 시작한다.

### 리스크 6 — 약어의 의미 모호성
- **문제**: `DP`, `RLCR` 같은 약어는 문맥마다 의미가 다르다.
- **완화**: skeleton 단계에서는 bucket만 잡고, runnable 전환 시 acronym clarification section을 필수화한다.

## 20. 구현 전 확인할 후속 질문

이 스펙은 1차 구조 확장을 위한 것이므로, 실제 implementation plan 단계에서 아래를 구체화해야 한다.

1. `02_deep_learning`의 어느 unit부터 runnable 승격을 시작할 것인가?
2. `05_advanced_nlp_llm`에서 acronym ambiguity(`DP`, `RLCR`)를 어떤 공식 용어로 고정할 것인가?
3. agent role docs는 `.codex/` 아래 prompt/skill 형태로 둘지, `docs/agents/` 설명 문서로 둘지?
4. status metadata를 `lesson.yaml`에 둘지 별도 manifest에 둘지?
5. multimodal renumbering 시 기존 링크/migration note를 어떤 방식으로 제공할지?

## 21. 최종 권고

이번 단계는 **코어 사다리를 깨지 않고 전체 지도를 다시 넓히는 재정렬 작업**으로 정의해야 한다. 가장 좋은 방향은 다음과 같다.

1. `00_foundations`는 얇고 공통적인 기초로 유지한다.
2. `01_ml`는 실험 discipline 트랙으로 유지한다.
3. `02_deep_learning`를 대형 신규 코어 트랙으로 신설한다.
4. NLP를 `03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`의 3단 구조로 재정렬한다.
5. `06_training_systems`와 `07_frontier_labs`를 NLP 뒤에 둔다.
6. multimodal은 `08_multimodal_bridge -> 09_multimodal`로 후반부에 이동시킨다.
7. 인덱스만 늘리지 말고, 상태 모델 / 문서 계약 / 테스트 / 에이전트 역할 문서까지 함께 깐다.
8. 그 위에서 개별 unit를 점진적으로 runnable 상태로 승격한다.

이 방향이 사용자의 최신 의도를 가장 잘 반영하면서, BTB를 장기적으로 크게 확장할 수 있는 가장 안정적인 구조다.
