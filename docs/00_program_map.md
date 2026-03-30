# 00 Program Map

## 목표

BTB는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가면서, 각 단계에서 이론을 실험으로 검증하고 재현 가능한 산출물을 남기는 한글 중심 학습 사다리다.

## 왜 이 순서인가

1. `00_foundations` 에서 tensor, gradient, optimizer, GPU/runtime, attention 같은 공통 기초를 먼저 고정한다.
2. `01_ml` 에서 데이터 분할, metric, 에러 분석, 해석의 기본기를 익힌다.
3. `02_nlp_bridge` 에서 tokenization, embedding, sequence modeling, transformer block 감각을 연결한다.
4. `03_nlp` 에서 task-specific NLP 실습으로 확장한다.
5. `04_multimodal_bridge` 에서 alignment, retrieval vs generation, cross-attention 개념을 미리 연결한다.
6. `05_multimodal` 에서 modality alignment, 생성, reasoning까지 확장한다.

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
| ML | confusion matrix, residual plot, feature importance | 왜 이 모델이 이 데이터에서 강한가/약한가 |
| NLP | length distribution, class/entity/span error analysis | tokenizer와 pretraining이 무엇을 해결하는가 |
| Multimodal | retrieval grid, caption panel, QA failure panel | 모델이 두 modality를 정말 함께 쓰는가 |

## 추천 진행 방식

1. 각 stage에서 가장 쉬운 데이터셋으로 빠른 1차 실험
2. 같은 stage에서 더 어려운 데이터셋으로 확장
3. baseline과 강한 baseline을 모두 남기기
4. 숫자만 저장하지 말고 figure와 failure case를 같이 저장
5. 같은 실수를 반복하지 않도록 `summary.md` 를 누적

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
