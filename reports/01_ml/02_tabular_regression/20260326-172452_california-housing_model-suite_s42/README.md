# 02. 표형 회귀 결과 요약

이 문서는 California Housing 회귀 실험을 **이론 → 실험 설계 → 결과 → 실패 분석 → 다음 가설** 순서로 정리한 학습 노트다.
회귀는 점수 하나만 보면 쉽게 오해한다. 그래서 이 문서는 숫자와 함께 **왜 그 숫자가 나왔는지**, **어떤 구간에서 틀렸는지**, **어떤 그림을 같이 봐야 하는지**를 함께 적는다.

## 한 줄 결론

- 과제: California Housing 회귀
- 최고 모델: `hist_gbdt`
- 핵심 지표: `rmse`=`0.4717`, `mae`=`0.3179`, `r2`=`0.8302`
- 해석: tree boosting이 선형 모델과 GPU MLP보다 더 잘 맞았고, 고가 주택과 특정 지역/소득 구간에서 residual이 크게 남았다.

## 이론 링크

- 상세 이론 문서: [THEORY.md](../../../../01_ml/02_tabular_regression/THEORY.md)

### 왜 이론을 먼저 봐야 하는가

회귀는 점수 하나만 보면 쉽게 착각한다.
MAE와 RMSE는 모두 오차를 보지만 서로 다른 리스크를 드러내고, R²는 평균 예측보다 얼마나 나은지 보여 주지만 tail error를 숨길 수 있다.
그래서 이 실험은 metric을 외우는 장이 아니라, **metric이 왜 생겼는지**와 **그 metric이 어떤 실패를 놓치는지**를 함께 보는 장이다.

## 핵심 이론 요약

- MAE는 평균 절대 오차다. 큰 오차가 전체를 과하게 흔들지 않아 해석이 쉽다.
- RMSE는 제곱 오차 기반이라 큰 오차를 더 강하게 벌한다.
- R²는 평균 예측 대비 설명력을 보여 준다.
- residual은 예측값과 실제값의 차이로, bias와 tail 실패를 찾는 핵심 도구다.
- parity plot, residual histogram, slice 분석을 함께 봐야 회귀를 제대로 해석할 수 있다.

## 왜 이런 이론이 생겼는가

회귀는 “얼마나 맞았는가”보다 “어디서 얼마나 틀렸는가”가 중요해지면서, 평균 오차 지표와 residual 진단이 함께 발전했다.
MAE는 이상치에 덜 민감한 직관적 지표로, RMSE는 큰 오차를 더 강하게 벌하는 위험 지표로, R²는 평균 기준선 대비 설명력을 보는 요약 지표로 자리 잡았다.

이번 실험에서는 이 세 관점을 모두 동시에 써야 했다.
평균 성능은 좋더라도, 고가 주택과 지역/소득 slice에서 구조적인 오차가 남았기 때문이다.

## 실험 설계

- 데이터셋: California Housing
- 분할: train / valid / test를 분리해서 같은 기준으로 비교
- 비교 모델:
  - `DummyRegressor(strategy="mean")`
  - `LinearRegression`
  - `Ridge`
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`
  - `GPU MLP`
- 공통 전처리:
  - 수치형 결측치 대체
  - linear/MLP 계열은 표준화
  - tree 계열은 결측치 대체만 사용

이 실험의 목적은 “어떤 모델이 최고인가”가 아니라, **어떤 가정을 가진 모델이 이 데이터의 구조를 가장 잘 설명하는가**를 확인하는 것이다.

## 모델 비교

| 모델 | RMSE | MAE | R2 | FIT_SEC |
| --- | --- | --- | --- | --- |
| hist_gbdt | 0.4717 | 0.3179 | 0.8302 | 1.56 |
| random_forest | 0.5146 | 0.3363 | 0.7979 | 0.52 |
| gpu_mlp | 0.5848 | 0.4032 | 0.7391 | 1.92 |
| ridge | 0.7329 | 0.5354 | 0.5901 | 0.04 |
| linear_regression | 0.7329 | 0.5354 | 0.5901 | 0.01 |
| dummy_mean | 1.1448 | 0.9031 | -0.0000 | 0.01 |

### 메트릭 해석

- `hist_gbdt`의 RMSE는 `dummy_mean` 대비 약 `58.8%` 낮다.
- `linear_regression` 대비 RMSE는 약 `35.6%` 낮다.
- `random_forest`보다도 약 `8.3%` 낮다.
- `gpu_mlp`보다도 약 `19.3%` 낮다.

즉, 이 데이터에서는 비선형 표형 모델이 분명히 필요하고, tree boosting이 가장 강한 기준선이다.

## 결과 해석 / 실패 분석

### 1. 전체적으로는 잘 맞지만 tail이 남는다

최악 예측 30개에서 평균 실제값은 `3.616`, 평균 예측값은 `2.489`였다.
이는 모델이 전반적으로 **과소추정** 경향이 있음을 의미한다.

### 2. 고가 주택을 충분히 따라가지 못한다

최악 사례 중 `target > 4.5` 비중이 `43.3%`였다.
즉, 상단 tail에서 오차가 많이 났다.
California Housing은 상단이 `5.0` 근처에서 막히는 데이터이므로, 이 구간은 모델이 특히 어렵다.

### 3. 저가 주택도 과대추정된다

최악 사례 중 `target < 1.0` 비중도 `13.3%`였다.
즉, 극단적으로 낮은 값도 평균 쪽으로 끌려 올라간다.

### 4. residual histogram은 약한 과소추정을 보여 준다

`residual_histogram.svg`를 보면 residual이 0 근처에 모여 있지만, `-0.1` 근처 bin이 가장 높다.
즉, 전체적으로는 괜찮아 보여도 약한 과소추정 경향이 남아 있다.

### 5. 지역별로도 오차가 다르다

`regional_error_slice.svg`에서는 위도 bucket이 낮은 쪽, 즉 남쪽에 가까운 구간의 MAE가 더 컸다.

- `(34.085, 35.63]`: `0.356`
- `(32.531, 34.085]`: `0.323`
- `(37.175, 38.72]`: `0.319`
- `(35.63, 37.175]`: `0.286`
- `(38.72, 40.265]`: `0.223`
- `(40.265, 41.81]`: `0.141`

전체 평균이 좋아도 지역별 난이도는 다를 수 있다.

### 6. 소득 구간별로도 오차가 다르다

`error_slice_by_income.svg`는 소득 bucket마다 RMSE가 달라진다는 것을 보여 준다.

- 최고 소득 bucket `(5.064, 15.0]`: `0.527`
- 중간 소득 bucket들: `0.490`, `0.463`, `0.451`
- 최저 소득 bucket `(0.499, 2.346]`: `0.421`

즉, 소득대에 따라 오차가 달라진다.
이것은 지역 slice와는 별개의 난이도 축이다.

### 7. feature importance는 위치와 소득이 핵심이다

Permutation importance에서 상위 feature는 다음 순서였다.

1. `Latitude` (`0.771`)
2. `Longitude` (`0.676`)
3. `MedInc` (`0.494`)
4. `AveOccup` (`0.185`)
5. `HouseAge` (`0.067`)
6. `AveRooms` (`0.056`)
7. `AveBedrms` (`0.011`)
8. `Population` (`0.009`)

즉, 이 문제는 단순 소득 회귀가 아니라 **위치 + 소득 + 주거 구조**가 함께 작동하는 문제다.

### 8. worst prediction cases는 평균 회귀 경향을 보여 준다

worst cases는 모델이 어려운 샘플에서 평균 쪽으로 끌리는 경향을 보여 준다.
고가 주택은 낮게, 저가 주택은 높게 예측하는 패턴이 함께 남아 있다.

## 결과 Figure

### parity_plot.svg

![](figures/results/parity_plot.svg)

예측값과 실제값이 대각선에서 얼마나 벗어나는지 본다.
고가 구간에서 아래로, 저가 구간에서 위로 치우치면 평균으로 끌리는 bias가 있다는 뜻이다.

### residual_histogram.svg

![](figures/results/residual_histogram.svg)

오차 분포가 0 근처에 모이는지, 한쪽 꼬리가 긴지 본다.
이번 실험에서는 약한 과소추정 경향이 남아 있다.

### residual_vs_target.svg

![](figures/results/residual_vs_target.svg)

target 값이 커질수록 residual이 어떻게 바뀌는지 본다.
tail에서 residual 폭이 커지고, 극단값에서 과소추정/과대추정이 함께 나타나는지 확인한다.

### learning_curve.svg

![](figures/results/learning_curve.svg)

학습 데이터가 늘어날수록 검증 오차가 어떻게 변하는지 본다.
데이터가 더 필요했는지, 아니면 모델 구조가 더 중요했는지 판단하는 데 쓴다.

### target_histogram.svg

![](figures/results/target_histogram.svg)

target 분포와 꼬리 구간을 본다.
California Housing은 tail이 강하고 상단 cap이 있어, 분포 자체가 단순하지 않다.

## 분석 Figure

### feature_importance.svg

![](figures/analysis/feature_importance.svg)

어떤 feature가 예측에 가장 크게 기여했는지 보여 준다.
여기서는 위치 변수와 `MedInc`가 핵심이었다.

### regional_error_slice.svg

![](figures/analysis/regional_error_slice.svg)

위도 구간별 MAE를 보여 준다.
전체 평균이 좋아도 지역별로 성능이 다를 수 있음을 확인시켜 준다.

### error_slice_by_income.svg

![](figures/analysis/error_slice_by_income.svg)

소득 구간별 RMSE 차이를 보여 준다.
지역 slice와는 다른 방향의 편차를 확인하는 데 유용하다.

### worst_prediction_cases.svg

![](figures/analysis/worst_prediction_cases.svg)

가장 크게 틀린 샘플을 보여 준다.
고가 주택을 낮게 예측하거나, 아주 낮은 가격을 높게 예측하는 패턴을 확인할 수 있다.

## 왜 이 결과가 나왔는가

- 선형 모델은 피처 간 비선형 상호작용과 공간 구조를 충분히 잡지 못했다.
- tree boosting은 구간별 규칙과 상호작용을 더 잘 잡았다.
- GPU MLP는 표현력은 있지만, tabular에서는 inductive bias가 tree보다 항상 유리하지 않았다.
- 타깃 cap과 지역성, 그리고 소득대별 차이 때문에 tail error가 남았다.

## 다음 실험 가설

1. `log1p(target)` 변환을 하면 tail 과소추정이 줄어드는가?
2. `Latitude`, `Longitude`를 지역 bin이나 상호작용 feature로 바꾸면 regional MAE가 줄어드는가?
3. Huber loss 또는 quantile regression을 쓰면 tail/이상치에 더 강해지는가?
4. neighborhood 수준의 공간 feature engineering을 추가하면 고가 주택 구간 성능이 나아지는가?

## 최신 자료 링크

- 최신 이론: [THEORY.md](../../../../01_ml/02_tabular_regression/THEORY.md)
- 최신 실행 입력: `runs/01_ml/02_tabular_regression/20260326-172452_california-housing_model-suite_s42/`
- 최신 원시 예측: `runs/01_ml/02_tabular_regression/20260326-172452_california-housing_model-suite_s42/predictions/worst_predictions.csv`

## 다음 단계

- 이론에서 배운 MAE/RMSE/R² 차이를 figure와 연결해서 읽는다.
- residual 구조를 설명할 수 있을 때까지 worst case를 다시 확인한다.
- 지역성, 소득대, tail 문제를 줄이는 후속 실험을 설계한다.
