# 03 Model Selection And Interpretation 이론

## 이 단계가 왜 따로 존재하는가

이 단계의 목적은 단순히 “점수를 잘 내는 모델”을 고르는 것이 아니다. 핵심은 **시간 순서가 있는 표형(tabular) 데이터에서 어떤 검증이 타당한지**, **어떤 모델이 배포 관점에서 안전한지**, **왜 그런 결론이 나왔는지 설명할 수 있는지**를 훈련하는 데 있다.

Bike Sharing 문제는 겉으로는 일반 회귀처럼 보이지만, 실제로는 다음 특성이 함께 섞여 있다.

- 시간 축이 존재한다: 과거 패턴이 미래 수요에 직접 영향을 준다.
- 수요 피크가 뚜렷하다: 출퇴근 시간대에 오차가 크게 벌어진다.
- 날씨와 요일이 강하게 상호작용한다: 같은 시간대라도 조건이 다르면 수요가 달라진다.
- 평균보다 tail과 peak가 중요하다: 운영에서는 “많이 틀렸는지”보다 “언제 틀렸는지”가 더 중요하다.

그래서 이 실습은 “아무 split으로나 학습한 뒤 숫자만 보는 일”과 “시간 구조를 보존하면서 모델을 선택하고 실패를 해석하는 일”의 차이를 몸에 익히는 데 초점을 둔다.

---

## 왜 time-aware validation 이론이 생겼는가

시간 데이터에 random split을 쓰면 미래 정보가 학습에 섞일 수 있다. 예를 들어 여름/겨울, 평일/주말, 출퇴근 시간대처럼 서로 다른 패턴이 랜덤하게 뒤섞이면 모델은 실제 배포 상황보다 훨씬 쉬운 문제를 풀게 된다. 이것이 **temporal leakage**다.

이런 문제를 막기 위해 `TimeSeriesSplit` 같은 시간-aware 검증이 널리 쓰이게 되었다. 이 접근은 다음을 보장하려고 한다.

- 과거 구간으로 학습한다.
- 미래에 가까운 구간으로 검증한다.
- 실제 운영처럼 “과거를 보고 미래를 예측”하는 구조를 만든다.

즉, time-aware validation은 단순히 CV를 바꾸는 기술이 아니라, **검증 절차 자체를 배포 조건에 맞추는 장치**다.

### time-aware validation이 해결하는 문제

- random split의 낙관적 편향을 줄인다.
- 시점이 달라졌을 때 성능이 얼마나 흔들리는지 볼 수 있다.
- 데이터 누수(leakage)로 인해 과대평가된 모델을 걸러낸다.
- 튜닝 결과가 “운 좋게 잘 맞은 split”에만 의존하지 않게 한다.

---

## leakage 이론이 왜 중요한가

`leakage`는 학습 시점에는 알 수 없어야 할 정보가 모델 입력이나 검증 구조에 들어가는 현상이다. 시간 데이터에서는 특히 다음 형태로 자주 나타난다.

- 미래 시점의 패턴이 train에 섞임
- target과 너무 가까운 후처리 결과가 feature처럼 들어감
- 전체 데이터 통계로 만든 전처리가 fold 경계를 무시함

leakage가 생기면 validation score는 좋아 보이지만, 실제 배포에서는 성능이 급락한다. 그래서 이 단계에서는 단순히 “좋은 모델”보다 **안전한 검증 절차**를 먼저 만들도록 훈련한다.

### leakage가 해결하는 문제

- 미래를 미리 본 것처럼 보이는 착시를 없앤다.
- 학습/검증 경계를 지켜서 진짜 일반화 성능을 측정하게 한다.
- 모델 선택 과정에서 생기는 자기기만을 줄인다.

---

## model selection 이론은 무엇을 해결하는가

모델 선택은 “점수가 가장 높은 모델 하나를 고르는 작업”보다 넓다. 실제로는 다음 질문을 동시에 다뤄야 한다.

- 평균 성능이 가장 좋은 모델은 무엇인가?
- fold별 분산이 작은가?
- 튜닝이 과하게 복잡하지 않은가?
- 해석이 가능한가?
- 운영 비용이 감당 가능한가?

즉, model selection은 **정확도 하나만 보지 않고 평균·분산·안정성·설명 가능성·비용을 함께 비교하는 절차**다.

### model selection이 해결하는 문제

- test 점수 하나로 모든 판단을 끝내는 실수를 막는다.
- 평균이 비슷해도 분산이 큰 모델을 경계하게 한다.
- 운영 리스크를 수치로 함께 읽게 한다.
- “왜 이 모델을 골랐는지”를 설명할 근거를 준다.

---

## interpretation 이론은 왜 필요한가

좋은 모델은 점수만 높고 끝나지 않는다. 왜 그런 결과가 나왔는지를 설명해야 한다. 특히 시간축이 있는 데이터는 평균 성능이 좋아도 특정 시간대, 특정 날씨, 특정 계절에서 크게 무너질 수 있다.

이런 이유로 permutation importance와 slice analysis가 필요하다.

### permutation importance

특정 feature를 섞었을 때 성능이 얼마나 떨어지는지를 본다. 이 값이 크면 그 feature가 모델 결정에 중요했다는 뜻이다. 내부 계수만 보는 방식보다 모델 종류에 덜 의존한다.

### slice analysis

전체 평균이 아니라 특정 구간에서 성능이 어떤지를 본다.

- 날씨가 나쁜 날
- 평일/주말
- 특정 시간대(출퇴근 peak)
- 특정 계절
- 높은 예측 구간

### interpretation이 해결하는 문제

- 평균 점수 뒤에 숨은 편향을 드러낸다.
- 모델이 무엇을 근거로 예측하는지 확인하게 한다.
- 다음 실험 가설을 더 정확하게 세우게 한다.

---

## 이번 실습의 문제 정의: count regression + 시간 패턴

`cnt`는 자전거 대여 수라는 **비음이 아닌 count형 목표값**이다. 일반적인 연속형 회귀처럼 다룰 수는 있지만, count 데이터는 다음 성질을 가진다.

- 0 이상이다.
- 분산이 평균보다 커질 수 있다.
- 특정 구간에서 급격한 peak가 발생한다.
- 큰 수요 예측 실패가 운영에 더 큰 영향을 준다.

따라서 이 문제는 단순 평균회귀보다 **peak와 tail을 함께 맞추는 문제**로 봐야 한다. 이 실습에서는 Poisson baseline, tuned HistGradientBoostingRegressor, GPU MLP를 함께 비교해서 어떤 모델이 시간 구조를 가장 잘 잡는지 본다.

---

## 실험이 실제로 어떻게 돌아가는가

### 1) 데이터와 split

- 대상 데이터: Bike Sharing Dataset
- 입력 특징: hour, season, workingday, weather, temp, hum, atemp 등
- 검증 방식: 시간 순서를 보존한 split + `TimeSeriesSplit`
- 최종 평가는 hold-out test window에서 수행

### 2) 비교 대상

- `poisson_baseline`: 최소 기준선
- `tuned_hist_gbdt`: 시간 패턴과 비선형 상호작용을 잘 잡는 후보
- `gpu_mlp`: 표현력은 크지만 tabular 구조에서는 불리할 수 있는 비교군

### 3) 튜닝과 선택

`HistGradientBoostingRegressor`의 복잡도를 `validation curve`와 CV fold score로 비교한다. 여기서 중요한 것은 단일 평균이 아니라 **평균 + 분산 + 안정성**이다.

### 4) 최종 해석

모델을 고른 뒤에는 permutation importance와 slice analysis로 “무엇을 근거로 예측했는지”, “어디서 무너졌는지”를 확인한다.

---

## 왜 이번 문제에서는 tree boosting이 강했는가

이론적으로 tabular + 시간 패턴 + 비선형 상호작용 조합에서는 tree boosting이 매우 자연스러운 선택이다.

- hour 같은 범주형/주기형 정보를 split으로 잘 쪼갤 수 있다.
- workingday, weather, temp 같은 feature 간 상호작용을 자연스럽게 포착한다.
- count peak처럼 국소적으로 튀는 구간을 MLP보다 더 안정적으로 잡는 경우가 많다.
- validation curve로 복잡도를 직접 조절하기 쉽다.

반대로 Poisson baseline은 구조가 너무 단순해서 peak와 상호작용을 충분히 잡지 못하고, GPU MLP는 학습은 되더라도 tabular 문제에서는 tree boosting만큼 안정적이지 않을 수 있다. 이 결과는 이론이 실제 실험과 잘 맞아떨어진 사례로 읽을 수 있다.

---

## 이번 리포트를 읽을 때 봐야 하는 아티팩트

- [최신 리포트 README](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/README.md)
- [CV fold RMSE boxplot](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/cv_fold_score_boxplot.svg)
- [validation curve](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/validation_curve.svg)
- [top feature importance](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/top_feature_importance.svg)
- [subgroup metric comparison](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/subgroup_metric_comparison.svg)
- [prediction-bin error plot](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/confidence_bin_plot.svg)
- [common failure slice summary](artifacts/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/common_failure_slice_summary.svg)

---

## 관찰된 failure slice를 어떻게 읽을 것인가

### 1) workingday × weather 조합에서 오차가 몰렸다

`common_failure_slice_summary.svg`를 보면 가장 어려운 구간은 다음과 같다.

| workingday | weather | MAE |
| --- | --- | --- |
| 0 | 3 | 73.30 |
| 1 | 3 | 64.58 |
| 0 | 2 | 44.63 |
| 0 | 1 | 38.36 |
| 1 | 2 | 37.60 |
| 1 | 1 | 32.04 |

여기서 `weather=3`은 불리한 날씨 범주로 해석할 수 있다. 즉, **날씨가 나빠질수록 오차가 커지고, workingday와의 조합에 따라 더 불안정해진다**는 뜻이다.

### 2) 출퇴근 시간대에서 peak miss가 발생했다

리포트 요약의 worst-case 분석에서는 오차가 다음 시간대에 집중되었다.

- 상위 시간대 분포: `{8: 9, 18: 4, 17: 4}`
- 상위 날씨 분포: `{3: 11, 1: 10, 2: 9}`
- 평균 실제 수요: `441.4`
- 평균 예측 수요: `401.6`

즉, 모델은 전체 평균은 잘 맞추지만 **출퇴근 peak를 낮게 보는 보수적 예측**을 하는 경향이 있다.

### 3) 예측값이 커질수록 오차가 커졌다

`prediction-bin error plot`은 높은 예측 구간일수록 MAE가 커지는 경향을 보여 준다. 이는 regression 버전의 confidence bin 해석으로 볼 수 있다. 예측값이 큰 구간이 곧 수요 peak에 가까우므로, **고수요 구간일수록 더 어렵다**는 뜻이다.

### 4) 계절별로도 난도가 달랐다

`subgroup metric comparison.svg`에서는 season code별 RMSE가 다르게 나타난다.

- `season_1`: `67.114`
- `season_4`: `63.290`
- `season_3`: `51.424`

즉, 전체 평균 RMSE 하나만 보면 보이지 않는 계절별 난이도 차이가 존재한다.

---

## feature importance가 말해 주는 것

`top feature importance.svg`는 모델이 실제로 무엇을 많이 사용했는지를 보여 준다.

상위 feature와 permutation importance drop은 다음과 같다.

- `hr`: `155.332`
- `hour_sin`: `65.230`
- `workingday`: `58.760`
- `hour_cos`: `37.046`
- `temp`: `36.313`
- `hum`: `17.997`
- `atemp`: `11.591`
- `day_of_year`: `9.100`
- `weathersit`: `8.035`
- `weekday`: `7.004`

이 패턴은 모델이 **시간대 정보와 주기성, 작업일 여부, 날씨/기온**을 핵심 신호로 사용하고 있음을 보여 준다. 즉, 이번 문제는 “아무 feature나 넣으면 되는 회귀”가 아니라 **시간-기상-캘린더 상호작용을 어떻게 설명하느냐**가 중요하다.

---

## 최종적으로 무엇을 기억해야 하는가

- `TimeSeriesSplit`은 시간 데이터에서 leakage를 줄이기 위한 검증 장치다.
- leakage는 validation을 망가뜨리고 모델 선택을 왜곡한다.
- model selection은 평균 점수뿐 아니라 분산, 안정성, 운영 리스크를 함께 본다.
- interpretation은 평균 뒤에 숨은 failure slice를 드러낸다.
- 이번 실습에서는 tuned HGBDT가 가장 좋은 평균 성능과 안정성을 보였지만, 악천후와 출퇴근 peak에서는 여전히 약점이 남아 있다.

---

## 다음 실험 가설

1. lag feature를 추가하면 peak 수요 예측이 좋아지는가?
2. holiday / working day / weather interaction을 더 명시적으로 만들면 악천후 slice가 개선되는가?
3. target log transform 또는 다른 count-friendly loss를 쓰면 tail error가 줄어드는가?
4. 출퇴근 시간대만 별도 모델로 분리하면 peak miss가 줄어드는가?
5. fold 수를 늘리면 validation 안정성이 더 선명하게 보이는가?

이 단계에서 진짜로 배워야 하는 것은 “어떤 모델이 이겼는가”가 아니라, **왜 이 모델이 선택되었고, 어디서 틀렸으며, 다음에 무엇을 실험해야 하는가**를 설명하는 능력이다.
