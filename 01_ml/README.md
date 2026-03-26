# 01 ML

이 트랙의 목표는 `표형 데이터 -> 전처리 -> 베이스라인 -> 강한 베이스라인 -> 평가 -> 해석` 흐름을 몸에 익히는 것이다.

딥러닝으로 바로 들어가기 전에, 아래를 확실히 익힌다.

- train/valid/test 분리
- 누수 방지
- 수치형/범주형 전처리
- 적절한 평가 지표 선택
- 교차검증과 모델 선택
- 에러 분석과 해석 가능성

## 전용 conda 환경

ML 트랙은 다른 단계와 의존성이 충돌할 수 있으므로 전용 환경 `btb-01-ml` 에서 실행한다.

- 최소 환경 스펙: [env/environment.yml](env/environment.yml)
- 실제 실행 환경 lock: [env/conda-linux-64.lock.txt](env/conda-linux-64.lock.txt)
- 환경/실행 가이드: [env/README.md](env/README.md)
- 환경 생성 스크립트: `bash 01_ml/env/create_env.sh`
- 전체 실행 진입점: `bash 01_ml/run_ml_track.sh`

## 코드 배치

stage별로 바로 찾아볼 수 있게 실행 코드를 각 폴더 안에 함께 둔다.

- `01_tabular_classification/run_stage.py`
- `02_tabular_regression/run_stage.py`
- `03_model_selection_and_interpretation/run_stage.py`
- `04_large_scale_tabular/run_stage.py`
- 공통 유틸/리포트 빌더: `01_ml/common.py`, `01_ml/reporting.py`

## 이론 문서

- 트랙 공통 이론: [THEORY.md](THEORY.md)
- 각 stage별 이론은 각 폴더의 `THEORY.md`를 본다.

이번 프로젝트에서 실제로 쓸 확정 데이터셋 표는 [00_dataset_assignments.md](00_dataset_assignments.md) 에 고정했다.

## 단계 구성

| Stage | 목적 | 추천 데이터셋 | 약한 베이스라인 | 강한 베이스라인 | 남길 figure |
| --- | --- | --- | --- | --- | --- |
| [01_tabular_classification](01_tabular_classification/README.md) | 분류 기본기 | `scikit-learn/adult-census-income` | Dummy, Logistic Regression | Random Forest, GBDT | ROC/PR, confusion matrix, calibration |
| [02_tabular_regression](02_tabular_regression/README.md) | 회귀와 residual 분석 | `California Housing` | Linear Regression, Ridge | Random Forest Regressor, GBDT | parity plot, residual plot, feature importance |
| [03_model_selection_and_interpretation](03_model_selection_and_interpretation/README.md) | CV, HPO, 해석 | `Bike Sharing Dataset` | untuned baseline | tuned GBDT / ensemble | CV boxplot, slice metrics, permutation importance |
| [04_large_scale_tabular](04_large_scale_tabular/README.md) | 서버 학습 전 scale-up | `mstz/covertype` | linear / shallow tree | histogram GBDT / boosted tree pipeline | throughput, memory, metric-vs-cost |

## 이번 프로젝트 기준 데이터셋 라인업

| Dataset | 어느 step에서 쓰는가 | 규모/형태 | 왜 지금 단계에 적합한가 | 공식 출처 |
| --- | --- | --- | --- | --- |
| `scikit-learn/adult-census-income` | Step 01 | 32k+ rows, mixed-type binary classification | 범주형/수치형 혼합 분류에서 전처리 파이프라인을 익히기 좋다 | https://huggingface.co/datasets/scikit-learn/adult-census-income |
| `California Housing` | Step 02 | 20,640 samples, regression | 회귀 실험과 residual 분석을 빠르게 돌리기 좋다 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html |
| `Bike Sharing Dataset` | Step 03 | 17,389 rows, count regression | `TimeSeriesSplit` 과 leakage 방지를 배우기 좋다 | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| `mstz/covertype` | Step 04 | 581k+ rows급 multiclass tabular | 서버 학습 전에 scale-up과 비용 측정을 해보기 좋다 | https://huggingface.co/datasets/mstz/covertype |
| `HIGGS` | Step 04 이후 확장 | 11M rows, binary classification | 서버 전용 대형 benchmark | https://archive.ics.uci.edu/dataset/280/higgs |

## 이 트랙에서 꼭 남길 것

- feature 분포 요약
- 결측치/범주형 처리 방식
- baseline 대비 개선폭
- calibration 여부
- 어떤 샘플에서 반복적으로 틀리는지

## 선택형 확장

- `Breast Cancer Wisconsin`: 코드 디버그용 초소형 classification
- `Ames Housing`: feature engineering 중심의 추가 회귀 실습
- `HIGGS`: 서버에서 다뤄야 하는 최종 대규모 tabular benchmark

실험 운영 규칙은 [../docs/01_experiment_playbook.md](../docs/01_experiment_playbook.md) 를 따른다.
