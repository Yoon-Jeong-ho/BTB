# 03 Model Selection And Interpretation

## 이 단계의 핵심 질문

이 단계는 단순히 점수를 높이는 실습이 아니다. **시간축이 있는 표형 데이터에서 어떤 검증이 타당한지**, **어떤 모델이 운영 관점에서 안전한지**, **왜 그런 결론이 나왔는지 설명할 수 있는지**를 배우는 단계다.

학습 포인트는 다음 네 가지다.

1. random split이 왜 위험한지 이해한다.
2. validation 전략이 모델 선택을 어떻게 바꾸는지 본다.
3. 평균 점수만이 아니라 분산과 실패 slice를 함께 읽는다.
4. 결과를 다음 실험 가설로 연결한다.

---

## 왜 이론이 필요한가

시간 데이터에서 random split을 쓰면 미래 정보가 학습에 섞일 수 있다. 여름/겨울, 평일/주말, 출퇴근 시간대처럼 서로 다른 패턴이 랜덤하게 뒤섞이면 모델은 실제 배포 상황보다 훨씬 쉬운 문제를 풀게 된다. 이것이 temporal leakage다.

그래서 이 단계에서는 `TimeSeriesSplit`, leakage 방지, model selection, interpretation이 모두 함께 등장한다. 각각이 해결하는 문제는 다르지만 목적은 같다. **검증 결과를 실제 운영과 더 가깝게 만드는 것**이다.

- `TimeSeriesSplit`: 과거로 학습하고 미래로 검증하게 해 leakage를 줄인다.
- leakage 방지: 미래를 미리 본 것 같은 착시를 막는다.
- model selection: 평균, 분산, 안정성, 해석 가능성을 함께 비교한다.
- interpretation: 평균 뒤에 숨어 있는 실패 slice를 드러낸다.

더 자세한 배경과 개념 정리는 [THEORY.md](THEORY.md)를 먼저 읽는 것이 좋다.

---

## 이번 프로젝트 기준 확정 데이터셋

- 실행 코드: `run_stage.py`
- 이론 문서: [THEORY.md](THEORY.md)
- 최신 리포트: [20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/README.md](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/README.md)

- Primary dataset: `Bike Sharing Dataset`
- Source: UCI Machine Learning Repository
- Load:

```python
from ucimlrepo import fetch_ucirepo

bike = fetch_ucirepo(id=275)
X = bike.data.features
y = bike.data.targets
```

- 왜 이 데이터인가: 같은 tabular라도 시간 축이 들어오면 split 전략, validation 방식, 해석 포인트가 완전히 달라지기 때문이다.
- 핵심 포인트: `TimeSeriesSplit`, validation strategy, tuning cost, slice analysis

> 참고: 위 로드 코드는 실제 실습에서 사용되는 데이터 로딩 형태를 보여 주는 메모다. `ucimlrepo`를 사용할 때는 문서와 환경 버전에 맞게 호출 이름을 확인해야 한다.

---

## 이 데이터에서 꼭 기억할 것

- `cnt`는 count형 목표값이다. peak를 놓치면 오차가 크게 벌어진다.
- 날씨와 working day는 시간대와 강하게 상호작용한다.
- random split은 미래 정보를 섞을 수 있으므로 실제 운영 감각을 왜곡한다.
- 평균 RMSE가 좋아도 출퇴근 peak나 악천후 slice에서 무너지면 실무적으로는 위험하다.

---

## 실험 파이프라인

1. 시간 순서를 유지하는 train/test split을 만든다.
2. Poisson baseline으로 최소 기준을 둔다.
3. `HistGradientBoostingRegressor` 후보를 `TimeSeriesSplit`으로 비교한다.
4. `validation curve`로 복잡도 방향을 확인한다.
5. GPU MLP를 비교군으로 추가한다.
6. 최종 best model을 test window에서 평가한다.
7. permutation importance와 slice analysis로 실패를 해석한다.

---

## 이번 실험에서 본 모델과 지표

| 모델 | RMSE | MAE | R2 | FIT_SEC |
| --- | --- | --- | --- | --- |
| tuned_hist_gbdt | 60.0516 | 38.1593 | 0.9258 | - |
| poisson_baseline | 163.1814 | 120.1109 | 0.4522 | - |
| gpu_mlp | 164.3160 | 109.0213 | 0.4446 | - |

### 지표 읽기 메모

- `RMSE 60.05`: peak miss에 벌점이 크게 들어가는 문제 구조를 보여 준다.
- `MAE 38.16`: 평균적으로 한 시간 예측이 어느 정도 흔들리는지 보여 준다.
- `R2 0.9258`: 전체 변동 설명력은 높지만, 특정 시간대/날씨 조합의 실패를 숨기지는 못한다.

즉, 이 문제는 R² 하나만 높다고 끝나는 문제가 아니다. **peak를 얼마나 놓치는지, slice별로 어떤 패턴이 남는지까지 같이 봐야 한다.**

---

## 왜 tuned HGBDT가 선택되었나

`tuned_hist_gbdt`는 CV에서 평균 성능과 안정성이 가장 좋았고, test window에서도 가장 낮은 RMSE를 기록했다. CV 후보의 흐름은 다음과 같다.

| 후보 | params | mean_rmse | std_rmse |
| --- | --- | --- | --- |
| candidate_1 | `learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, max_iter=160` | 67.6508 | 12.4027 |
| candidate_2 | `learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, max_iter=220` | 70.4737 | 15.1867 |
| candidate_3 | `learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=30, max_iter=280` | 74.1568 | 18.9147 |

이 결과는 **복잡도가 무조건 높다고 좋은 것이 아니고, 시간-aware CV에서 평균과 분산이 함께 좋아야 한다**는 점을 보여 준다.

---

## 이론과 실험을 연결해서 읽는 법

- `TimeSeriesSplit`은 validation 점수의 신뢰도를 높이기 위한 장치다.
- `validation curve`는 과소적합/과적합의 균형을 보는 장치다.
- `permutation importance`는 모델이 무엇을 근거로 예측하는지 확인하는 장치다.
- `slice analysis`는 평균 점수 뒤의 실패를 드러내는 장치다.

이 네 개를 같이 봐야 model selection이 단순한 점수 경쟁이 아니라는 점이 보인다.

---

## 이번 리포트에서 바로 읽어야 할 그림

- [CV fold RMSE boxplot](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/cv_fold_score_boxplot.svg)
- [validation curve](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/validation_curve.svg)
- [top feature importance](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/top_feature_importance.svg)
- [subgroup metric comparison](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/subgroup_metric_comparison.svg)
- [prediction-bin error plot](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/confidence_bin_plot.svg)
- [common failure slice summary](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/common_failure_slice_summary.svg)

---

## 승격 기준

다음 질문에 답할 수 있으면 이 단계를 제대로 이해한 것이다.

- 왜 random split이 위험한지 설명할 수 있는가?
- 왜 `TimeSeriesSplit`이 더 타당한지 설명할 수 있는가?
- 왜 tuned HGBDT가 선택되었는지 평균과 분산까지 말할 수 있는가?
- 어디서 틀렸는지 slice 단위로 설명할 수 있는가?
- 다음 실험이 어떤 가설을 검증해야 하는지 말할 수 있는가?

---

## 읽기 순서

1. [THEORY.md](THEORY.md)
2. [최신 리포트 README](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/README.md)
3. 아래 그림들을 순서대로 확인한다.

---

## 그림 바로가기

- [CV fold RMSE boxplot](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/cv_fold_score_boxplot.svg)
- [validation curve](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/validation_curve.svg)
- [top feature importance](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/results/top_feature_importance.svg)
- [subgroup metric comparison](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/subgroup_metric_comparison.svg)
- [prediction-bin error plot](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/confidence_bin_plot.svg)
- [common failure slice summary](../../reports/01_ml/03_model_selection_and_interpretation/20260326-172503_bike-sharing-hourly_tuned-hgbdt_s42/figures/analysis/common_failure_slice_summary.svg)
