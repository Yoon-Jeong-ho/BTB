# 02 표형 회귀 이론

이 장은 “회귀 모델이 맞다/틀리다”를 넘어서, **왜 이런 지표가 생겼고 어떤 실패를 잡기 위해 만들어졌는지**를 이해하는 노트다.
분류는 정답 라벨만 맞추면 되지만, 회귀는 연속값의 **오차 크기**, **오차 방향**, **오차가 생기는 구간**, **오차가 구조를 가지는지 여부**까지 봐야 한다.

California Housing 실습이 좋은 이유는 다음과 같다.

- 타깃이 연속값이라 오차의 크기와 방향을 직접 볼 수 있다.
- `MedInc`, `Latitude`, `Longitude`처럼 의미가 분명한 feature가 있어 해석이 쉽다.
- 집값은 지역성과 비선형성이 강해서 선형 모델의 한계가 잘 드러난다.
- 타깃이 `5.0` 근처에서 상단이 막히는 구조라서 tail error와 saturation bias를 보기 좋다.

즉, 이 장은 **오차 정의 → 지표 선택 → residual 진단 → slice 분석 → 다음 가설**의 순서로 회귀를 읽는 법을 정리한다.

## 1. 회귀 이론은 왜 생겼는가

회귀 이론은 “값을 하나 맞추면 끝”이라는 단순한 관점만으로는 실제 문제를 설명할 수 없어서 생겼다.
연속값 예측에서는 아래 질문이 모두 중요하다.

1. 평균적으로 얼마나 틀리는가?
2. 큰 실수가 얼마나 위험한가?
3. 특정 구간에서만 계속 틀리는가?
4. 단순 평균 예측보다 얼마나 나은가?
5. 오차가 우연한 잡음인지, 아니면 구조적인 편향인지?

이 질문에 답하기 위해 loss, metric, residual analysis가 분리되어 발전했다.

## 2. 회귀가 해결하려는 문제

회귀는 입력 `x`에 대해 실수형 출력 `ŷ = f(x)`를 추정한다.
그런데 실무에서 중요한 것은 “숫자를 하나 뱉었다”가 아니라 아래를 아는 것이다.

- 전체적으로 얼마나 맞는가
- 극단값에서 얼마나 무너지는가
- 특정 지역이나 특정 구간에서 편향이 생기는가
- 기준선보다 얼마나 좋은가
- 모델의 구조가 데이터의 구조와 맞는가

이때부터 지표와 residual이 필요해진다.

## 3. 지표가 분화된 이유

평균 오차만 보면 중요한 실패를 놓칠 수 있다.
그래서 회귀에서는 서로 다른 질문에 답하는 지표들이 따로 생겼다.

### 3.1 MAE: “평균적으로 얼마나 틀리는가”

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE는 절대 오차의 평균이다.
해석이 직관적이어서 “평균적으로 몇 단위 틀리는지”를 바로 읽을 수 있다.

이 지표가 필요한 이유:

- 큰 오차 하나에 전체 점수가 과하게 흔들리지 않는다.
- 이상치가 있어도 비교적 안정적이다.
- 운영 관점에서 가장 먼저 묻는 질문인 “얼마나 틀리나”에 직접 답한다.

즉, MAE는 **평균적인 예측 품질**을 보는 지표다.

### 3.2 RMSE: “큰 실수가 얼마나 위험한가”

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE는 제곱 오차의 제곱근이다.
큰 오차를 더 강하게 벌주기 때문에 tail error에 민감하다.

이 지표가 필요한 이유:

- 드물지만 큰 실패를 줄이고 싶을 때 유용하다.
- 오차가 정규분포적이라고 보면 least squares와 연결된다.
- 실무에서 “가끔 크게 틀리는 문제”를 더 잘 드러낸다.

즉, RMSE는 **리스크 중심 지표**다.

### 3.3 R²: “평균 예측보다 얼마나 나은가”

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

R²는 모델이 단순 평균 예측보다 얼마나 나은지 보여 준다.

- 0에 가까우면 평균 예측과 비슷하다.
- 1에 가까울수록 설명력이 높다.
- 음수면 평균 예측보다도 못할 수 있다.

다만 R²가 높아도 모든 구간을 잘 맞춘다는 뜻은 아니다.
평균적으로는 좋아도 고가 구간이나 특정 지역에서 계속 틀릴 수 있다.
그래서 R²는 MAE/RMSE와 함께 읽어야 한다.

## 4. 기준선은 왜 평균 예측인가

회귀에서는 가장 단순한 기준선으로 상수 예측을 쓴다.
어떤 상수가 최적인지는 loss에 따라 달라진다.

- MSE 기준 최적 상수는 평균값
- MAE 기준 최적 상수는 중앙값

이번 실습에서 `DummyRegressor(strategy="mean")`를 쓴 이유는, squared-loss 관점에서 평균값 예측이 가장 자연스러운 하한선이기 때문이다.
이 기준선이 있어야 모델이 “평균만 찍는 수준”을 얼마나 넘어섰는지 읽을 수 있다.

이번 run에서 dummy mean의 RMSE는 `1.1448`이었다.
이 값은 다른 모델이 넘어야 할 출발점이다.

## 5. residual 분석은 왜 생겼는가

### 5.1 residual의 정의

$$
\text{residual}_i = \hat{y}_i - y_i
$$

residual은 예측값과 실제값의 차이다.

- residual > 0: 과대추정
- residual < 0: 과소추정

중요한 것은 평균 residual이 0에 가깝냐보다, **어떤 구간에서 residual이 한쪽으로 치우치느냐**다.

### 5.2 residual 분석이 해결하는 문제

평균 지표만 보면 모델의 구조적 실패를 놓친다.
residual을 보면 다음을 알 수 있다.

- 특정 target 구간에서만 반복적으로 과소/과대추정하는가
- 분산이 target에 따라 달라지는가
- tail에서 오차가 급격히 커지는가
- 지역 slice에서 편향이 생기는가

즉, residual 분석은 **오차의 구조를 읽는 도구**다.
회귀 이론이 발전하면서 residual을 보는 습관이 생긴 이유도 여기에 있다.

## 6. 각 그림이 해결하는 문제

### 6.1 parity plot

parity plot은 `true target`과 `prediction`을 같은 축에 놓고 본다.
완벽하면 대각선 위에 있어야 한다.

이 그림이 답하는 질문:

- 전체적으로 위로 뜨는가, 아래로 깔리는가
- 고값 구간을 과소추정하는가
- 저값 구간을 과대추정하는가
- target이 커질수록 산포가 커지는가

### 6.2 residual histogram

residual histogram은 오차 분포를 본다.
0 근처에 모여 있으면 전체적으로는 잘 맞는 것이고, 한쪽 꼬리가 길면 그 방향의 편향이 남아 있다는 뜻이다.

### 6.3 residual vs target

residual vs target은 target 크기에 따라 오차 구조가 어떻게 바뀌는지 보여 준다.
이 그림이 중요한 이유는 **“값이 커질수록 더 어려워지는가”**를 직접 보기 때문이다.

### 6.4 target histogram

target histogram은 데이터 분포를 먼저 보여 준다.
타깃이 한쪽으로 치우치거나 cap이 있으면 모델이 평균으로 끌리는 현상이 쉽게 생긴다.

### 6.5 feature importance

feature importance는 무엇이 모델을 움직였는지 보여 준다.
이는 “모델이 어디를 보고 예측했는가”를 해석하는 단서다.

### 6.6 regional error slice

regional slice는 전체 평균 아래 숨은 지역 차이를 드러낸다.
회귀는 전체 점수만 좋아도 특정 지역에서 계속 틀릴 수 있으므로, slice가 필요하다.

### 6.7 error slice by income bucket

소득 구간별 slice는 “어떤 소득대에서 더 어렵나”를 본다.
지역 slice가 공간 차이를 보여 준다면, income slice는 경제 수준에 따른 난이도 차이를 보여 준다.

### 6.8 worst prediction cases

worst cases는 모델이 어떤 상황에서 무너지는지 보여 준다.
한두 개 샘플이 아니라 실패 패턴을 확인하는 데 쓰인다.

## 7. California Housing 데이터가 이 이론에 잘 맞는 이유

이 데이터는 단순 숫자 회귀가 아니라 지리·소득·주거 구조가 섞인 문제다.

- `MedInc`: 지역 소득 수준의 강한 신호
- `Latitude`, `Longitude`: 공간 구조
- `HouseAge`, `AveRooms`, `AveOccup`, `AveBedrms`: 주거 패턴의 압축 정보

특히 집값은 해안/내륙, 남/북, 소득대에 따라 다르게 움직인다.
그래서 선형 관계만으로는 부족하고, residual과 slice가 중요해진다.

또한 타깃이 `5.0` 근처에서 상단이 막히는 구조가 있어, 고가 주택을 더 이상 세밀하게 구분하지 못하는 saturation이 생긴다.
이런 구조는 tail error를 매우 잘 드러낸다.

## 8. 이번 실험의 모델 순서는 왜 이렇게 잡았는가

실습은 쉬운 기준선에서 강한 모델로 올라가도록 구성했다.

1. `DummyRegressor`
   - 평균 예측 기준선을 만든다.
2. `LinearRegression`, `Ridge`
   - 선형 가정이 충분한지 확인한다.
3. `RandomForestRegressor`
   - 비선형성과 feature interaction을 본다.
4. `HistGradientBoostingRegressor`
   - tabular strong baseline의 힘을 본다.
5. `GPU MLP`
   - GPU를 쓴 신경망이 tabular에서 항상 이기지 않는다는 점을 확인한다.

이 순서는 “최고 모델 찾기”가 아니라 **어떤 가정을 어떤 순서로 깨보는가**를 보여 준다.

## 9. 이번 실험 결과를 읽는 법

이번 run의 핵심 결과는 다음과 같다.

- `hist_gbdt`: RMSE `0.4717`, MAE `0.3179`, R² `0.8302`
- `random_forest`: RMSE `0.5146`, MAE `0.3363`, R² `0.7979`
- `gpu_mlp`: RMSE `0.5848`, MAE `0.4032`, R² `0.7391`
- `linear_regression` / `ridge`: RMSE `0.7329`, MAE `0.5354`, R² `0.5901`
- `dummy_mean`: RMSE `1.1448`, MAE `0.9031`, R² `≈ 0`

여기서 읽어야 할 점은 세 가지다.

1. 선형 모델만으로는 부족하다.
   ridge와 linear regression이 거의 같다는 것은 정규화보다 표현력 한계가 더 중요하다는 뜻이다.
2. tree boosting이 가장 강하다.
   RMSE는 dummy 대비 약 58.8%, linear regression 대비 약 35.6% 낮다.
3. GPU MLP는 GPU를 쓴다고 자동으로 이기지 않는다.
   tabular에서는 inductive bias가 중요하고, 구조화된 데이터에는 tree 계열이 여전히 강하다.

## 10. 아티팩트에서 실제로 보인 실패 패턴

### 10.1 평균으로 끌리는 경향

최악 예측 30개에서 평균 실제값은 `3.616`, 평균 예측값은 `2.489`였다.
이는 모델이 전체적으로는 잘 맞아도, 어려운 샘플에서 아래쪽으로 끌리는 경향이 있음을 보여 준다.

### 10.2 고가 주택 과소추정

최악 사례 중 `target > 4.5` 비중이 `43.3%`였다.
상단 cap이 있는 데이터에서 고가 구간은 특히 어렵다.
이 구간은 모델이 평균 쪽으로 수축되는지 확인하는 데 가장 중요하다.

### 10.3 저가 주택 과대추정

최악 사례 중 `target < 1.0` 비중도 `13.3%`였다.
즉, 아주 낮은 가격도 평균 쪽으로 끌어올리는 과대추정이 남아 있다.

### 10.4 지역 slice의 해석

`regional_error_slice.svg`에서는 위도 bucket이 낮은 쪽, 즉 남쪽에 가까운 구간의 MAE가 더 컸다.

- `(34.085, 35.63]`: `0.356`
- `(32.531, 34.085]`: `0.323`
- `(37.175, 38.72]`: `0.319`
- `(35.63, 37.175]`: `0.286`
- `(38.72, 40.265]`: `0.223`
- `(40.265, 41.81]`: `0.141`

전체 평균이 좋아도 지역별 난이도는 다를 수 있다.

### 10.5 소득 slice의 해석

`error_slice_by_income.svg`는 소득 bucket마다 RMSE가 달라진다는 것을 보여 준다.

- 최고 소득 bucket `(5.064, 15.0]`: `0.527`
- 중간 소득 bucket들: `0.490`, `0.463`, `0.451`
- 최저 소득 bucket `(0.499, 2.346]`: `0.421`

즉, 소득대에 따라 오차가 달라진다.
이것은 지역 slice와는 별개의 난이도 축이다.

### 10.6 feature importance의 해석

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

## 11. 이론과 실험을 함께 읽으면 남는 결론

- 회귀는 평균 오차만 보면 안 되고 residual 구조를 봐야 한다.
- MAE와 RMSE는 같은 방향의 숫자가 아니라 서로 다른 실패 리스크를 보여 준다.
- R²는 평균 예측보다 나은지 보여 주지만 tail failure를 숨길 수 있다.
- 지역/소득 slice를 봐야 평균 점수 아래 숨은 편차를 찾을 수 있다.
- tabular 문제에서는 GPU 신경망이 항상 강하지 않고, tree 기반 strong baseline이 매우 강하다.

## 12. 다음 실험 가설

다음 실험에서 검증할 가설은 다음과 같다.

1. `log1p(target)` 변환을 하면 tail 과소추정이 줄어드는가?
2. `Latitude`, `Longitude`를 지역 bin이나 상호작용 feature로 바꾸면 slice MAE가 줄어드는가?
3. Huber loss나 quantile regression을 쓰면 outlier와 tail에 더 강해지는가?
4. neighborhood 수준의 spatial feature engineering을 넣으면 고가 주택 구간이 개선되는가?

## 13. 이 장을 읽는 순서

1. [README.md](README.md)에서 실험 전체 흐름을 먼저 본다.
2. 아래 최신 보고서에서 실제 수치를 확인한다.
3. 결과 그림을 보면서 residual, slice, feature importance를 함께 읽는다.

### 최신 보고서와 그림

- 최신 결과 보고서: [README](artifacts/20260327-164513_california-housing_model-suite_s42/README.md)
- parity plot: [parity_plot.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/results/parity_plot.svg)
- residual histogram: [residual_histogram.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/results/residual_histogram.svg)
- residual vs target: [residual_vs_target.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/results/residual_vs_target.svg)
- feature importance: [feature_importance.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/analysis/feature_importance.svg)
- regional slice: [regional_error_slice.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/analysis/regional_error_slice.svg)
- income slice: [error_slice_by_income.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/analysis/error_slice_by_income.svg)
- worst cases: [worst_prediction_cases.svg](artifacts/20260327-164513_california-housing_model-suite_s42/figures/analysis/worst_prediction_cases.svg)
