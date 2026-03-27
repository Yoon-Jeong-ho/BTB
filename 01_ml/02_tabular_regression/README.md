# 02 표형 회귀

이 장은 **회귀 지표를 읽는 법**과 **오차 구조를 해석하는 법**을 익히는 학습 노트다.
단순히 모델을 한 번 돌리는 장이 아니라, **왜 이런 지표가 생겼는지**, **각 지표가 어떤 문제를 해결하는지**, **어떤 구간에서 계속 틀리는지**를 함께 본다.

## 학습 목표

- 회귀에서 loss와 metric이 왜 분리되는지 이해한다.
- MAE, RMSE, R²가 각각 어떤 질문에 답하는지 구분한다.
- residual과 parity plot으로 bias를 읽는 법을 익힌다.
- feature importance와 slice 분석으로 실패 구간을 찾는다.
- 선형 모델, tree 모델, GPU MLP를 같은 기준으로 비교한다.

## 왜 이 장이 필요한가

분류는 맞고 틀림이 분명하지만, 회귀는 다르다.
회귀에서는 다음을 같이 읽어야 한다.

- 평균적으로 얼마나 틀리는가
- 큰 오차가 어디서 생기는가
- 특정 구간에서만 계속 틀리는가
- 단순 평균 예측보다 얼마나 나은가
- 오차가 구조를 가지는가

그래서 이 장은 숫자를 외우는 장이 아니라, **오차를 설명하는 언어를 배우는 장**이다.

## 왜 California Housing인가

이 데이터는 회귀를 공부하기에 매우 좋다.

- 타깃이 연속값이라 오차를 직접 해석할 수 있다.
- `MedInc`, `Latitude`, `Longitude` 같은 해석 가능한 feature가 있다.
- 지역성과 비선형성이 강해서 선형 모델의 한계가 잘 드러난다.
- 타깃이 `5.0` 근처에서 막히는 구조가 있어 tail error와 saturation bias를 보기 좋다.

## 읽는 순서

1. 먼저 [THEORY.md](THEORY.md)를 읽는다.
2. 아래 실습 파이프라인으로 어떤 모델을 어떤 순서로 비교하는지 확인한다.
3. 최신 결과는 [report](../../reports/01_ml/02_tabular_regression/20260326-172452_california-housing_model-suite_s42/README.md)에서 본다.
4. 마지막으로 figure와 worst case를 함께 읽는다.

## 이번 프로젝트 기준 확정 데이터셋

- 실행 코드: `run_stage.py`
- 이론 문서: [THEORY.md](THEORY.md)

- Primary: `California Housing`
- Source: scikit-learn builtin dataset
- Load:

```python
from sklearn.datasets import fetch_california_housing

frame = fetch_california_housing(as_frame=True)
df = frame.frame
```

- 왜 지금 단계에 맞는가: 설치 직후 바로 불러올 수 있고, 회귀 metric과 residual 분석을 빠르게 반복하기 좋다.
- Extension: `Ames Housing`

## 실습 파이프라인

1. target 분포와 이상치를 먼저 확인한다.
2. `DummyRegressor`로 평균값 기준선을 만든다.
3. `LinearRegression`, `Ridge`로 선형 가정을 시험한다.
4. `RandomForestRegressor`와 `HistGradientBoostingRegressor`로 비선형성과 interaction을 본다.
5. `GPU MLP`로 “GPU를 쓰는 신경망이 tabular에서 항상 더 좋은가?”를 확인한다.
6. MAE, RMSE, R²를 함께 비교한다.
7. residual을 target 구간별로 분석한다.
8. feature importance와 regional / income slice로 실패 구조를 찾는다.

## 결과를 읽을 때 봐야 할 것

- RMSE가 가장 낮은 모델이 무엇인지
- dummy mean 대비 얼마나 개선됐는지
- 고가 주택 구간에서 과소추정이 남아 있는지
- latitude / longitude slice에서 오차가 커지는지
- 소득 구간별 RMSE가 달라지는지
- GPU MLP가 tree model보다 실제로 나은지

## 결과로 남길 figure

| Figure | 읽는 포인트 |
| --- | --- |
| `parity_plot.svg` | 예측값과 실제값이 대각선에서 얼마나 벗어나는지 본다. |
| `learning_curve.svg` | 학습 데이터가 늘어날수록 검증 오차가 어떻게 변하는지 본다. |
| `residual_histogram.svg` | 오차 분포가 중앙에 모이는지, 꼬리가 긴지 본다. |
| `residual_vs_target.svg` | 타깃 크기에 따라 residual이 어떻게 달라지는지 본다. |
| `target_histogram.svg` | 타깃 분포와 꼬리 구간을 본다. |

## 분석으로 남길 figure

| Figure | 읽는 포인트 |
| --- | --- |
| `feature_importance.svg` | 어떤 feature가 예측에 가장 크게 기여하는지 본다. |
| `regional_error_slice.svg` | 위도 구간별로 MAE가 어떻게 달라지는지 본다. |
| `error_slice_by_income.svg` | 소득 구간별 RMSE 차이를 본다. |
| `worst_prediction_cases.svg` | 가장 크게 틀린 샘플이 어떤 패턴인지 본다. |

## 현재 실험 요약

최신 실험에서는 `HistGradientBoostingRegressor`가 가장 좋았다.

- RMSE: `0.4717`
- MAE: `0.3179`
- R²: `0.8302`

기준선과 비교하면 다음처럼 읽을 수 있다.

- `dummy_mean` RMSE `1.1448`보다 크게 좋아졌다.
- `linear_regression` / `ridge`의 RMSE `0.7329`보다도 더 낮다.
- `random_forest`보다도 더 낮다.
- `gpu_mlp`보다도 더 좋았기 때문에, 이 데이터에서는 tree 계열의 inductive bias가 유리하다는 해석이 가능하다.

## 최신 리포트

- [최신 결과 보고서](../../reports/01_ml/02_tabular_regression/20260326-172452_california-housing_model-suite_s42/README.md)

## 핵심 해석 메모

- 전체 평균이 좋아도 residual과 slice를 보면 tail과 지역성이 남는다.
- MAE는 평균 오차를, RMSE는 큰 오차 리스크를, R²는 평균 예측 대비 설명력을 보여 준다.
- 회귀 실험은 숫자 하나가 아니라, **오차 구조 전체**를 읽는 연습이다.

## 승격 기준

- residual 구조를 설명할 수 있다.
- target 구간별 성능 차이를 말할 수 있다.
- 단순 평균 예측보다 충분히 나아진다.
