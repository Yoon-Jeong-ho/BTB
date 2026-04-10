# 00 Program Map

## 목표

BTB는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가면서, 각 단계에서 이론을 실험으로 검증하고 재현 가능한 산출물을 남기는 한글 중심 학습 사다리다.

## 왜 이 순서인가

1. `00_foundations` 에서 tensor, activation, loss, gradient, optimizer, GPU/runtime 같은 공통 기초를 먼저 고정한다.
2. `01_ml` 에서 데이터 분할, metric, 에러 분석, 해석의 기본기를 익힌다.
3. `02_nlp_bridge` 에서 tokenization, embedding, attention, transformer block 감각을 연결한다.
4. `03_nlp` 에서 task-specific NLP 실습으로 확장한다.
5. `04_multimodal_bridge` 에서 alignment, retrieval vs generation, cross-attention 개념을 미리 연결한다.
6. `05_multimodal` 에서 retrieval, generation, reasoning까지 확장한다.

## 현재 학습 가능 상태

- `00_foundations`: 공통 기초 5 unit가 모두 채워져 있다.
- `01_ml`: 표형 데이터 기준의 applied ML baseline 트랙이 준비돼 있다.
- `02_nlp_bridge -> 03_nlp`: bridge 2 unit 뒤에 applied NLP 3 unit가 이어진다.
- `04_multimodal_bridge -> 05_multimodal`: bridge 1 unit 뒤에 applied multimodal 3 unit가 이어진다.

즉 지금은 **00→05 전체를 끊기지 않게 공부할 수 있는 상태**다.

## 딥러닝 코어 축

딥러닝 기초를 먼저 굳히고 싶다면 아래 축을 한 묶음으로 본다.

1. `00_foundations/01_tensor_shapes`
2. `00_foundations/02_activation_and_loss`
3. `00_foundations/03_gradients_and_backpropagation`
4. `00_foundations/04_regularization_and_normalization`
5. `00_foundations/05_gpu_memory_runtime`
6. `02_nlp_bridge/01_tokenization_and_embeddings`
7. `02_nlp_bridge/02_attention_and_transformer_block`

이 축은 activation, loss, gradient, backprop, masking, attention, runtime을 하나의 숫자 흐름으로 이해하게 만드는 데 목적이 있다. `04_multimodal_bridge/01_contrastive_alignment` 는 이 다음에 붙이는 선택 확장이다.

## 한 unit씩 공부할 큰 흐름

세부 unit-by-unit 순서는 [02_study_guide.md](02_study_guide.md) 를 따른다. 여기서는 track 수준의 큰 흐름만 잡는다.

1. `00_foundations` — tensor부터 gradient, regularization, runtime까지 DL 코어를 먼저 고정한다.
2. `01_ml` — baseline, metric, interpretation, failure analysis로 실험 discipline을 붙인다.
3. `02_nlp_bridge` — tokenization, embedding, attention, transformer block을 연결한다.
4. `03_nlp` — text classification, NER, MRC로 applied NLP를 읽는다.
5. `04_multimodal_bridge` — alignment와 retrieval vs generation을 미리 연결한다.
6. `05_multimodal` — retrieval, captioning, VQA로 멀티모달을 확장한다.

## 단계 구조

1. `00_foundations`: 모든 상위 트랙이 공유하는 기초 개념과 runtime 감각
2. `01_ml`: 실험 discipline과 해석 기본기
3. `02_nlp_bridge`: ML에서 NLP로 넘어가기 위한 표현 학습 브리지
4. `03_nlp`: 본격 NLP 실습 트랙
5. `04_multimodal_bridge`: NLP에서 멀티모달로 넘어가기 위한 브리지
6. `05_multimodal`: 본격 멀티모달 실습 트랙

## 단계별 산출물

| Track | 반드시 남길 것 | 핵심 질문 |
| --- | --- | --- |
| Foundations | activation/loss figure, gradient check, runtime observation | 모델 안의 숫자 흐름이 어디서 바뀌는가 |
| ML | confusion matrix, residual plot, feature importance | 왜 이 모델이 이 데이터에서 강한가/약한가 |
| NLP | length distribution, class/entity/span error analysis | tokenizer와 pretraining이 무엇을 해결하는가 |
| Multimodal | retrieval grid, caption panel, QA failure panel | 모델이 두 modality를 정말 함께 쓰는가 |

## 추천 진행 방식

1. 각 unit에서 먼저 README와 THEORY를 읽는다.
2. 그다음 `scratch_lab.py` 또는 대표 figure를 보고 숫자 흐름을 확인한다.
3. `framework_lab.py`와 실행 결과 예시를 보며 실제 라이브러리 구현과 연결한다.
4. `analysis.md`에서 왜 이런 결과가 나왔는지 읽는다.
5. 마지막에 `reflection.md`로 스스로 설명 가능한지 점검한다.

## 서버와 로컬의 역할 분리

- 로컬: 자료 조사, config 작성, 결과 정리, figure 선별, 보고서 작성
- 서버: 실제 학습, sweep, 대형 로그/체크포인트 생성

## 언어 정책

- README, THEORY, PREREQS, 분석/회고 문서는 한글 우선을 기본으로 한다.
- 필요한 경우 영어 technical term를 병기하되, 설명 문장은 한국어 중심으로 유지한다.

## 최소 성공 기준

- 각 stage마다 최소 1개 데이터셋에서 end-to-end 재현
- `reports/` 에 승격된 실험 보고서 존재
- 중요한 모델은 `artifacts/MODEL_REGISTRY.md` 에 기록
