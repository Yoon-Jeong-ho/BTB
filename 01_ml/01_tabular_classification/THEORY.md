# 01 Tabular Classification 이론

## 왜 이 실습을 하는가

Adult 데이터는 범주형 + 수치형 + class imbalance가 함께 들어 있어, 표형 분류의 기본기를 익히기 좋다.

## 핵심 이론

### 1. Classification threshold

확률 예측은 점수이고, `0.5` 같은 threshold를 두는 순간 label이 된다.
따라서 AUROC/AUPRC와 F1은 서로 다른 질문에 답한다.

### 2. AUROC vs AUPRC

- AUROC: 전체 ranking 품질
- AUPRC: positive가 적을 때 실제 운영 감각에 더 가깝다

이 데이터에서는 `>50K` 가 소수 클래스라서 AUPRC 해석이 중요하다.

### 3. Calibration

예측 확률 `0.9` 라고 했을 때 실제로 90% 맞는지 보는 것이 calibration이다.
실무에서는 ranking뿐 아니라 calibration도 중요하다.

### 4. Confusion matrix 해석

- FP: 실제 <=50K인데 >50K로 예측
- FN: 실제 >50K인데 <=50K로 예측

둘 중 어떤 오류가 더 비싼지는 실제 서비스 정책이 결정한다.

## 분석 포인트

- 어떤 demographic / education slice에서 오차가 높아지는가
- 고확신 오답이 어떤 feature 패턴에서 나오는가
- 확률이 높을수록 실제 accuracy도 같이 오르는가
