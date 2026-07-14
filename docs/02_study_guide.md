# 02 Study Guide

## 목적

이 문서는 BTB의 `00→10` 커리큘럼을 어떻게 읽고 들어갈지 정리한 한국어 우선 학습 가이드다. 현재 `docs/curriculum_status.json`에 선언된 전체 unit은 `runnable` 상태다. 그래도 **실행 전에 manifest를 canonical source of truth로 확인하고, 각 track README의 status table을 supplementary context로 참고**해야 한다.

현재 학습 경로와 인덱싱은 `00→10`이 기준이다. 과거 경로명은 migration note에서만 historical reference로 다룬다. 시작 전에는 [learner preflight](00_learner_preflight.md) (`docs/00_learner_preflight.md`)에서 Python/CLI, 수학, 확률/metric, PyTorch/GPU 준비도를 확인한다.

## 용어 정리

- **Track**: `00_foundations`, `01_ml`처럼 큰 학습 구간이다.
- **Unit**: track 안의 개별 학습 폴더다. 표준 unit은 README/THEORY/실습 코드/analysis/reflection 흐름을 따른다.
- **Stage**: `01_ml`에서 쓰는 legacy 이름이다. 웹사이트에서는 unit처럼 다루되 `dataset.py`, `experiment.py`, `run_stage.py`를 함께 읽는다.
- **Bridge 문서**: 다음 개념 세계로 넘어가기 전 읽는 연결 문서다. 항상 별도 실습 파일이 있는 것은 아니지만, 다음 unit의 선행 체크 역할을 한다.
- **Runnable**: 실행 명령과 산출물 위치가 있는 학습 항목이다. 표준 lesson 실습, ML stage runner, GPU 선택 실험처럼 실행 계약은 서로 다를 수 있다.

## 표준 1-pass 루트

폴더의 canonical 전체 순서는 아래처럼 그대로 유지한다.

`00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal -> 10_vla`

다만 초심자 **core path**는 `00→05 -> 08_multimodal_bridge -> 09_multimodal -> 10_vla`로 진행한다. `06_training_systems`와 `07_frontier_labs`는 각각 분산/GPU 운영 또는 연구 재현/capstone이 필요할 때 돌아오는 **선택형 사이드카**다.

이 루트는 아래 상황에 적합하다.

- 기초 수학/텐서 감각부터 LLM·멀티모달까지 한 번에 지도처럼 보고 싶은 경우
- 모델 family와 applied task의 경계를 먼저 잡고, 이후 training system/frontier experiment를 선택하려는 경우
- 당장 모든 unit를 실행하지 않더라도 전체 프로그램의 역할 분리를 먼저 이해하고 싶은 경우

### 1-pass에서 보는 법

1. `00_foundations`에서 tensor, gradient, runtime 관측 습관을 먼저 만든다.
2. `01_ml`에서 baseline, metric, failure analysis를 실험 discipline으로 굳힌다.
3. `02_deep_learning`에 들어가기 전 [feature matrix to neural training bridge](04_feature_matrix_to_neural_training_bridge.md)로 `fit/predict` 감각이 PyTorch training loop로 어떻게 바뀌는지 확인한다.
4. `02_deep_learning`에서 딥러닝 코어인 perceptron·CNN·RNN·transformer·generative model family를 지도처럼 훑는다.
5. `03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`에서 NLP/LLM 흐름을 연결한다.
6. `06_training_systems`와 `07_frontier_labs`는 큰 실험 운영과 재현을 위한 선택형 사이드카다. GPU·분산·논문 재현이 당장 필요 없으면 core path 완료 뒤로 미룬다.
7. `08_multimodal_bridge -> 09_multimodal`에서 image-text shared representation과 응용 태스크로 넘어간다.
8. `10_vla`에서 multimodal understanding을 action token과 safety gate로 연결한다.

## 무기초 → LLM / RLHF / Multimodal / VLA 루트

LLM과 VLA까지 목표라면 아래 체크포인트를 빠뜨리지 않는다.

1. `00_foundations/01_tensor_shapes`, `02_activation_and_loss`, `03_gradients_and_backpropagation`으로 shape/loss/update 언어를 만든다.
2. `01_ml`에서 baseline, metric, error analysis를 먼저 익히고, [feature matrix to neural training bridge](04_feature_matrix_to_neural_training_bridge.md)로 딥러닝 training loop와 연결한다.
3. `02_deep_learning/04_attention_and_transformers`와 `03_nlp_bridge/02_attention_and_transformer_block`으로 attention을 숫자 흐름으로 설명한다.
4. `04_nlp` 전체를 통해 tokenizer/encoder/task head를 applied task에서 확인한다.
5. `05_advanced_nlp_llm/01~05`로 pretraining objective, data mixture, DAPT, SFT, preference optimization을 본다. 들어가기 전에 [decoder generation bridge](06_decoder_generation_bridge.md) (`docs/06_decoder_generation_bridge.md`)로 autoregressive decoding, temperature/top-p, prompt serialization, KV-cache 감각을 먼저 만든다.
6. RLHF 전에 [RL primer](05_rl_primer_for_rlhf.md)를 읽고 reward/policy/rollout/advantage/KL/PPO 용어를 정리한다.
7. `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`에서 RLHF/reasoning RL을 본다.
8. `08_multimodal_bridge -> 09_multimodal`로 image-text retrieval/captioning/VQA를 실행한다. retrieval에서 captioning/VQA로 넘어가기 전 [multimodal generation bridge](07_multimodal_generation_bridge.md) (`docs/07_multimodal_generation_bridge.md`)로 cross-attention, fusion, grounding failure를 확인한다.
9. `10_vla/01_vision_language_action_grounding` 전에 [RL to VLA bridge](08_rl_to_vla_bridge.md) (`docs/08_rl_to_vla_bridge.md`)로 MDP, trajectory, behavior cloning, offline RL, action space design을 읽고 action token과 safety gate를 확인한다.
10. `06_training_systems`와 `07_frontier_labs`는 남는 GPU, 분산 학습, 논문 재현, capstone sandbox가 필요해지는 시점에 나중 선택 구간으로 되돌아온다.

이 경로를 따르면 LLM base가 없는 학습자도 LLM, RLHF, multimodal, VLA 입구까지 같은 산출물 규칙으로 이어갈 수 있다. `10_vla/01_vision_language_action_grounding`은 VLA grounding entry point로, 실제 로봇 제어 전체가 아니라 multimodal 이해를 행동 토큰과 safety gate에 연결하는 최소 실험이다. 다만 일반 제어 RL 전체가 필요하면 별도 RL 교과 과정이 추가로 필요하므로, BTB 안에서는 [RL to VLA bridge](08_rl_to_vla_bridge.md) (`docs/08_rl_to_vla_bridge.md`)를 “최소 연결 다리”로 먼저 읽는다.

## 개념 브리지 문서

학습자가 특정 트랙에서 갑자기 어려워지는 지점을 줄이기 위해 아래 bridge 문서를 먼저 읽는다.

- [Decoder generation bridge](06_decoder_generation_bridge.md) (`docs/06_decoder_generation_bridge.md`): `04_nlp`의 encoder/task-head 감각에서 `05_advanced_nlp_llm`의 autoregressive decoder, sampling, temperature, top-k/top-p, prompt serialization, KV-cache로 넘어가는 다리.
- [Feature matrix to neural training bridge](04_feature_matrix_to_neural_training_bridge.md) (`docs/04_feature_matrix_to_neural_training_bridge.md`): `01_ml`의 feature matrix, baseline, metric discipline을 `02_deep_learning`의 tensor batch, training loop, learned representation으로 옮기는 다리.
- [Multimodal generation bridge](07_multimodal_generation_bridge.md) (`docs/07_multimodal_generation_bridge.md`): retrieval-style shared embedding에서 cross-attention, encoder-decoder captioning, VQA fusion, grounding failure로 넘어가는 다리.
- [RL to VLA bridge](08_rl_to_vla_bridge.md) (`docs/08_rl_to_vla_bridge.md`): RLHF reward/policy 언어와 VLA의 MDP, trajectory, behavior cloning, offline RL, action space design을 구분하는 다리.

## 딥러닝 코어 우선 루트

딥러닝 코어만 먼저 단단히 잡고 싶다면 아래 unit를 우선 실행한다.

1. `00_foundations/02_activation_and_loss` — logits, activation, loss가 학습 신호로 바뀌는 최소 감각
2. `00_foundations/03_gradients_and_backpropagation` — gradient flow와 update 방향 감각
3. `02_deep_learning/01_perceptron_and_mlp` — 가장 작은 supervised neural model
4. `02_deep_learning/04_attention_and_transformers` — attention/transformer core
5. `03_nlp_bridge/02_attention_and_transformer_block` — text 입력 관점의 attention bridge

이 루트는 `04_nlp`나 `05_advanced_nlp_llm`로 넘어가기 전에 모델 내부의 shape, signal, update, attention 흐름을 먼저 설명할 수 있게 만드는 압축 경로다.

## NLP / LLM 집중 압축 루트

NLP·LLM 중심으로 빠르게 올라가고 싶다면 아래 압축 루트를 권장한다.

`00_foundations -> 01_ml -> 02_deep_learning(핵심 일부) -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`

### 왜 `02_deep_learning`을 완전히 건너뛰지 않는가

- NLP에 바로 들어가더라도 perceptron/MLP, sequence model, attention/transformer 계열은 한 번은 봐야 한다.
- 최소한 `02_deep_learning`의 핵심 일부를 통해 neural architecture family와 training/debugging 감각을 먼저 잡아 두면 `03_nlp_bridge`와 `04_nlp`의 실험이 훨씬 덜 추상적으로 보인다.
- 이후 LLM 단위에서 pretraining objective, tokenizer/data mixture, instruction tuning을 이해할 때도 도움이 된다.

### NLP / LLM 압축 루트에서 우선순위

- `02_deep_learning`: transformer/sequence model/training recipe 중심으로 핵심 일부를 먼저 본다.
- `03_nlp_bridge`: tokenization, embedding, attention, mask를 반드시 확인한다.
- `04_nlp`: text classification -> NER -> MRC 순으로 applied NLP core를 익힌다.
- `05_advanced_nlp_llm`: pretraining 이후 objective, post-training, RAG, alignment를 차례로 연결한다.

## planned 상태를 읽는 법

- 현재 manifest의 모든 unit은 `runnable`이다.
- 향후 새 unit가 추가되어 `planned`나 `outlined`가 다시 등장하면, 그 상태는 아직 설계/문서 중심 단계로 이해한다.
- `runnable`이라고 표시된 unit만 바로 실행 실습 대상으로 기대하는 것이 안전하다.

## runnable lab를 기대하기 전에 확인할 것

1. [docs/curriculum_status.json](curriculum_status.json)에서 최신 `planned` / `runnable` 상태를 먼저 확인한다.
2. track README에 unit table이나 status note가 있으면 보조 설명으로 함께 읽는다.
3. 해당 unit 폴더에 `lesson.yaml`, lab 스크립트, artifact scaffold가 실제로 있는지 본다.
4. 실행 전에 [scripts/README.md](../scripts/README.md)와 [docs/01_experiment_playbook.md](01_experiment_playbook.md)로 산출물 규약을 확인한다.

## 추천 학습 운영 팁

- 전체를 빨리 훑고 싶다면 각 track README만 먼저 읽고, runnable unit만 골라 실습한다.
- 처음부터 frontier track까지 모두 돌리려 하지 말고, 관심 분야에 따라 압축 루트를 선택한다.
- 어떤 루트를 타더라도 `summary.md`, failure case, figure를 남기는 실험 습관은 공통으로 유지한다.
