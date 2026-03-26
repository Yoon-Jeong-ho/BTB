# 01 Tabular Classification

## 목표

표형 분류 문제에서 `누수 없는 전처리 + baseline 비교 + 혼동행렬 해석` 을 익힌다.

## 추천 데이터셋

- `Breast Cancer Wisconsin (Diagnostic)`: 가장 빠른 입문용
- `Adult`: 실제형 전처리와 범주형 처리까지 포함한 본격 실습용

## 실습 파이프라인

1. 데이터 카드 작성
2. train/valid/test split과 seed 고정
3. 수치형/범주형 컬럼 분리
4. `ColumnTransformer` 기반 전처리 파이프라인 구축
5. `DummyClassifier` 와 `LogisticRegression` 으로 baseline 생성
6. `RandomForest` 또는 `HistGradientBoosting` 으로 강한 baseline 생성
7. AUROC, AUPRC, F1, calibration 비교
8. 실패 샘플과 confusion pair 분석

## 결과로 남길 figure

- 클래스 분포
- 주요 feature histogram
- ROC curve / PR curve
- confusion matrix
- calibration curve

## 분석으로 남길 figure

- permutation importance
- error slice plot
- confidence vs correctness plot

## 승격 기준

- baseline 대비 개선이 분명하다.
- 데이터 누수 가능성이 제거되었다.
- 결과 figure와 분석 figure만 봐도 왜 성능이 달라졌는지 설명 가능하다.
