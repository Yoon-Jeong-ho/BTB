# 02 Study Guide

## 목적

이 문서는 BTB의 `00→09` 커리큘럼을 어떻게 읽고 들어갈지 정리한 한국어 우선 학습 가이드다. BTB는 전체 사다리를 먼저 공개한 상태이므로, 새 scaffold track이나 unit는 아직 `planned`일 수 있다. 따라서 **실행 가능한 lab를 기대하기 전에는 각 track README와 `docs/curriculum_status.json`의 unit status를 먼저 확인**해야 한다.

## 표준 1-pass 루트

가장 권장하는 기본 루트는 전체 계단을 순서대로 한 번 통과하는 방식이다.

`00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal`

이 루트는 아래 상황에 적합하다.

- 기초 수학/텐서 감각부터 LLM·멀티모달까지 한 번에 지도처럼 보고 싶은 경우
- 모델 family, task, training system, frontier experiment 사이의 경계를 분명히 잡고 싶은 경우
- 당장 모든 unit를 실행하지 않더라도 전체 프로그램의 역할 분리를 먼저 이해하고 싶은 경우

### 1-pass에서 보는 법

1. `00_foundations`에서 tensor, gradient, runtime 관측 습관을 먼저 만든다.
2. `01_ml`에서 baseline, metric, failure analysis를 실험 discipline으로 굳힌다.
3. `02_deep_learning`에서 perceptron·CNN·RNN·transformer·generative model family를 지도처럼 훑는다.
4. `03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm`에서 NLP/LLM 흐름을 연결한다.
5. `06_training_systems -> 07_frontier_labs`에서 큰 실험을 운영하고 재현하는 법으로 확장한다.
6. `08_multimodal_bridge -> 09_multimodal`에서 image-text shared representation과 응용 태스크로 넘어간다.

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

- BTB의 새 scaffold track과 unit는 문서 구조가 먼저 열리고, 나중에 runnable lab가 채워질 수 있다.
- 따라서 `planned`라고 적혀 있다면 `아직 설계/문서 중심 단계`로 이해해야 한다.
- `runnable`이라고 표시된 unit만 바로 실행 실습 대상으로 기대하는 것이 안전하다.

## runnable lab를 기대하기 전에 확인할 것

1. track README에서 unit table의 `Status`를 먼저 본다.
2. [docs/curriculum_status.json](curriculum_status.json)에서 최신 `planned` / `runnable` 상태를 다시 확인한다.
3. 해당 unit 폴더에 `lesson.yaml`, lab 스크립트, artifact scaffold가 실제로 있는지 본다.
4. 실행 전에 [scripts/README.md](../scripts/README.md)와 [docs/01_experiment_playbook.md](01_experiment_playbook.md)로 산출물 규약을 확인한다.

## 추천 학습 운영 팁

- 전체를 빨리 훑고 싶다면 각 track README만 먼저 읽고, runnable unit만 골라 실습한다.
- 처음부터 frontier track까지 모두 돌리려 하지 말고, 관심 분야에 따라 압축 루트를 선택한다.
- 어떤 루트를 타더라도 `summary.md`, failure case, figure를 남기는 실험 습관은 공통으로 유지한다.
