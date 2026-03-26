# 00 Program Map

## 목표

`기초 ML -> NLP -> Multimodal` 순서로 올라가면서, 각 단계에서 이론을 실험으로 검증하고 재현 가능한 산출물을 남기는 것이다.

## 왜 이 순서인가

1. `ML` 에서 데이터 분할, metric, 에러 분석, 해석의 기본기를 먼저 익힌다.
2. `NLP` 에서 표현 학습, tokenizer, 전이학습, task-specific evaluation으로 확장한다.
3. `Multimodal` 에서 modality alignment, 생성, reasoning까지 확장한다.

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

## 최소 성공 기준

- 각 stage마다 최소 1개 데이터셋에서 end-to-end 재현
- `reports/` 에 승격된 실험 보고서 존재
- 중요한 모델은 `artifacts/MODEL_REGISTRY.md` 에 기록
