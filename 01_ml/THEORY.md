# 01 ML 이론 개요

ML 트랙의 목표는 단순히 모델을 돌리는 것이 아니라, **왜 이런 split/지표/모델/분석을 쓰는지 설명할 수 있는 상태**까지 가는 것이다.

## 1. 데이터 분할 이론

- `train / valid / test` 는 역할이 다르다.
  - `train`: 파라미터 학습
  - `valid`: 모델 선택 / threshold / 하이퍼파라미터 선택
  - `test`: 최종 1회 평가
- test를 여러 번 보며 의사결정하면 test leakage가 생긴다.
- 시간축이 있는 데이터는 random split보다 **time-aware split** 이 우선이다.

## 2. 전처리 이론

- 수치형은 결측치 대체와 스케일링 여부를 모델에 맞게 결정한다.
- 범주형은 one-hot/target encoding 같은 표현 방식이 필요하다.
- 전처리는 반드시 train에서 fit하고 valid/test에는 transform만 해야 한다.
- 파이프라인을 쓰는 이유는 **누수 방지**다.

## 3. 평가 지표 이론

- classification에서는 imbalance가 있으면 accuracy만 보면 안 된다.
  - AUROC: 전반적인 ranking 품질
  - AUPRC: positive class가 희소할 때 더 실용적
  - F1: threshold 기반 precision-recall 균형
- regression에서는 RMSE와 MAE를 같이 본다.
  - RMSE: 큰 오차에 더 민감
  - MAE: 해석이 직관적
  - R²: 설명력 요약
- multiclass에서는 accuracy 외에 **macro-F1 / macro-recall** 로 class별 균형을 본다.

## 4. 모델 이론

- 선형 모델은 빠르고 해석이 쉽지만 비선형 상호작용 표현이 약하다.
- tree ensemble은 tabular에서 강력한 baseline이다.
- gradient boosting은 tabular strong baseline으로 자주 쓰인다.
- neural network는 GPU를 쓰기 쉽지만, tabular에서는 항상 tree보다 좋은 것은 아니다.

## 5. 분석 이론

실험은 점수표로 끝나면 안 된다. 최소한 아래 질문에 답해야 한다.

1. 어떤 데이터 구간에서 자주 틀리는가?
2. 모델이 어떤 feature에 의존하는가?
3. calibration은 괜찮은가?
4. residual/error가 구조적으로 남는가?
5. 다음 실험 가설은 무엇인가?

## 6. ML 트랙에서 꼭 익혀야 하는 사고 방식

- 점수가 좋아졌다고 바로 믿지 않는다.
- split이 타당한지 먼저 본다.
- baseline 대비 개선폭을 본다.
- figure를 통해 점수 차이를 설명한다.
- failure case를 보고 다음 실험을 설계한다.
