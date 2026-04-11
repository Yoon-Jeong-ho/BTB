# 00 Program Map

## 목표

BTB는 `00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal` 순서로 올라가면서, 각 단계에서 이론을 실험과 산출물로 검증하는 한글 우선 커리큘럼이다.

## 트랙 역할 경계

1. `00_foundations` — 공통 수치/텐서/실행 감각을 맞추는 진입 계단이다.
2. `01_ml` — 실험 discipline과 baseline 해석을 몸에 익히는 classical ML 구간이다.
3. `02_deep_learning` — perceptron, CNN, sequence model, transformer, generative model까지 딥러닝 모델 패밀리를 정리하는 구간이다.
4. `03_nlp_bridge` — 딥러닝에서 NLP로 넘어가는 입력 표현, tokenization, embedding, attention 감각을 연결하는 다리다.
5. `04_nlp` — text classification, NER, MRC 같은 applied NLP core를 실습하는 구간이다.
6. `05_advanced_nlp_llm` — pretraining 이후 고급 NLP/LLM, instruction tuning, preference optimization, RAG, alignment를 묶어 다루는 구간이다.
7. `06_training_systems` — distributed and large-model training systems를 다루는 운영/시스템 구간이다.
8. `07_frontier_labs` — reproduction, capstone, agentic experiments를 수행하는 개방형 연구 실습 구간이다.
9. `08_multimodal_bridge` — text-only 표현에서 image-text shared representation으로 넘어가는 multimodal 연결 다리다.
10. `09_multimodal` — retrieval, captioning, VQA 중심의 multimodal applied track이다.

## 왜 이 순서인가

1. `00_foundations`에서 tensor, gradient, optimizer, runtime을 먼저 고정해야 이후 모든 트랙의 실험 로그를 읽을 수 있다.
2. `01_ml`에서 baseline, metric, error analysis, reproducibility를 익혀야 이후 딥러닝/LLM 실험에서도 흔들리지 않는다.
3. `02_deep_learning`에서 모델 패밀리의 구조적 차이를 익혀 두면, NLP와 multimodal 실습에서 transformer를 블랙박스로 보지 않게 된다.
4. `03_nlp_bridge`는 문장이 token id, embedding, attention flow로 바뀌는 과정을 천천히 연결해 `04_nlp` 실습의 진입 장벽을 낮춘다.
5. `04_nlp`에서 task-specific NLP core를 경험한 뒤에야 `05_advanced_nlp_llm`의 pretraining/post-training 논의를 실제 문제와 연결할 수 있다.
6. `06_training_systems`는 큰 모델을 실제 하드웨어 위에서 운영하는 법을 분리해, 모델 설계와 시스템 설계를 동시에 혼동하지 않게 만든다.
7. `07_frontier_labs`는 앞선 트랙을 조합해 논문 재현, capstone, agentic workflow 실험으로 확장하는 sandbox다.
8. `08_multimodal_bridge`는 image-text alignment를 작은 예제로 먼저 익혀 `09_multimodal`의 retrieval/caption/VQA 실습으로 자연스럽게 이어 준다.

## 현재 학습 가능 상태

- 현재 [curriculum_status.json](curriculum_status.json)에 선언된 `00→09` 전체 unit은 `runnable` 상태다.
- `02_deep_learning`은 딥러닝 코어를, `03_nlp_bridge -> 04_nlp`는 NLP bridge/applied 흐름을, `08_multimodal_bridge -> 09_multimodal`은 multimodal bridge/applied 흐름을 담당한다.
- 과거 경로에서 현재 경로로 바뀐 대응표는 migration note의 historical reference로만 유지하며, 현재 실행 경로는 manifest와 track README를 canonical source of truth로 삼는다.

## 단계별 산출물 관점

| 구간 | 반드시 남길 것 | 핵심 질문 |
| --- | --- | --- |
| Foundations / ML | baseline metric, residual/error analysis, runtime 관측 | 기본기가 실제 실험 판단으로 이어지는가 |
| Deep Learning / NLP | shape trace, attention/sequence 분석, task failure case | 모델 내부 표현과 태스크 성능을 함께 설명할 수 있는가 |
| Advanced / Systems / Frontier | training log, system profile, ablation, self-review | 큰 모델·큰 실험을 운영 가능한 형태로 정리했는가 |
| Multimodal | retrieval grid, caption panel, VQA failure panel | 두 modality를 정말 함께 쓰는가 |

## 추천 진행 방식

1. 먼저 [docs/02_study_guide.md](02_study_guide.md)에서 표준 1-pass, 딥러닝 코어 우선, NLP/LLM 압축 루트 중 하나를 고른다.
2. `00_foundations -> 01_ml`을 끝내며 공통 실험 습관을 만든다.
3. 각 stage에서는 가장 쉬운 dataset/baseline으로 빠르게 1차 실험을 만든다.
4. 숫자만 저장하지 말고 figure, failure case, summary를 반드시 같이 남긴다.
5. 같은 실수를 반복하지 않도록 `summary.md`와 회고를 누적한다.

## 언어 정책

- README, THEORY, PREREQS, 분석/회고 문서는 한국어/한글 우선을 기본으로 한다.
- 필요한 경우 영어 technical term를 병기하되, 설명 문장은 한국어 중심으로 유지한다.
