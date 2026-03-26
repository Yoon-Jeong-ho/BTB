# 01 ML

이 트랙의 목표는 `표형 데이터 -> 전처리 -> 베이스라인 -> 강한 베이스라인 -> 평가 -> 해석` 흐름을 몸에 익히는 것이다.

딥러닝으로 바로 들어가기 전에, 아래를 확실히 익힌다.

- train/valid/test 분리
- 누수 방지
- 수치형/범주형 전처리
- 적절한 평가 지표 선택
- 교차검증과 모델 선택
- 에러 분석과 해석 가능성

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_tabular_classification](01_tabular_classification/README.md) | 분류 기본기 | Breast Cancer Wisconsin, Adult | Dummy, Logistic Regression | Random Forest, GBDT | ROC/PR, confusion matrix, calibration |
| [02_tabular_regression](02_tabular_regression/README.md) | 회귀와 residual 분석 | California Housing, Ames, Bike Sharing | Linear Regression, Ridge | Random Forest Regressor, GBDT | parity plot, residual plot, feature importance |
| [03_model_selection_and_interpretation](03_model_selection_and_interpretation/README.md) | CV, HPO, 해석 | Adult, Wine Quality, Covertype | untuned baseline | tuned GBDT / ensemble | CV boxplot, slice metrics, permutation importance |
| [04_large_scale_tabular](04_large_scale_tabular/README.md) | 서버 학습 전 scale-up | Covertype, HIGGS | linear / shallow tree | histogram GBDT / boosted tree pipeline | throughput, memory, metric-vs-cost |

## 추천 데이터셋

| Dataset | Task | 규모/형태 | 왜 좋은가 | 공식 출처 |
| --- | --- | --- | --- | --- |
| Adult | classification | 혼합형 tabular, 48,842 instances | 범주형/수치형 혼합, 불균형, 실제형 전처리 연습 | https://archive.ics.uci.edu/dataset/2/adult |
| Breast Cancer Wisconsin (Diagnostic) | classification | 569 instances, 30 features | 매우 빠르게 baseline과 해석을 연습 가능 | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| California Housing | regression | 20,640 samples, 8 features | 회귀 실험과 residual 분석에 적합 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html |
| Ames Housing | regression | 2,930 sales, 약 80 explanatory variables | feature engineering과 outlier 분석이 풍부하다 | https://jse.amstat.org/v19n3/decock.pdf |
| Bike Sharing | regression / count prediction | 17,389 rows, 13 features | 시계열 split과 leakage 방지 연습에 좋다 | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| Wine Quality | regression / ordinal classification | 4,898 white wine samples | 회귀와 분류 전환 실습 둘 다 가능 | https://archive.ics.uci.edu/dataset/186/wine+quality |
| Covertype | multiclass classification | 581,012 rows, 54 features | 서버 학습 전 대형 multiclass tabular benchmark로 좋다 | https://archive.ics.uci.edu/dataset/31/covertype |
| HIGGS | large-scale classification | 11,000,000 rows, 28 features | 대규모 ingestion, 저장소 분리, 비용 관리까지 연습 가능 | https://archive.ics.uci.edu/dataset/280/higgs |

## 이 트랙에서 꼭 남길 것

- feature 분포 요약
- 결측치/범주형 처리 방식
- baseline 대비 개선폭
- calibration 여부
- 어떤 샘플에서 반복적으로 틀리는지

## 선택형 확장

- `Ames Housing`: Boston 대체용 회귀 입문
- `Bike Sharing`: random split 대신 time-aware validation
- `Covertype`: multiclass와 tree ensemble scaling
- `HIGGS`: 서버에서 다뤄야 하는 대규모 tabular benchmark

실험 운영 규칙은 [../docs/01_experiment_playbook.md](../docs/01_experiment_playbook.md) 를 따른다.
