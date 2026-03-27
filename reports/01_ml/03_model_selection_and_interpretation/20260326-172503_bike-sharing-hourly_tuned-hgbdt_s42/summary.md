# 03. 모델 선택과 해석 요약 노트

## 한 줄 결론

- 과제: Bike Sharing 시간축이 있는 count 회귀
- 최고 모델: `tuned_hist_gbdt`
- 핵심 지표: `RMSE=60.0516`, `MAE=38.1593`, `R2=0.9258`
- 해석: 시간축을 보존한 validation이 이 문제에 맞는 방식이었고, tuned HGBDT가 peak 수요와 시간 패턴을 가장 안정적으로 잡았다.

---

## 이론 메모

자세한 배경은 [THEORY.md](../../../../01_ml/03_model_selection_and_interpretation/THEORY.md)를 먼저 보면 된다.

핵심은 네 가지다.

- `TimeSeriesSplit`: 과거로 학습하고 미래로 검증해서 leakage를 줄인다.
- leakage 방지: 미래를 미리 본 것 같은 착시를 막는다.
- model selection: 평균 점수, 분산, 안정성, 해석 가능성을 같이 본다.
- interpretation: 평균 뒤에 숨어 있는 failure slice를 드러낸다.

---

## 실험 설계

1. 시간 순서를 보존한 train/test split
2. Poisson baseline으로 최소 기준 마련
3. `HistGradientBoostingRegressor` 후보를 `TimeSeriesSplit`으로 비교
4. `validation curve`로 복잡도 방향 확인
5. GPU MLP를 비교 기준으로 추가
6. best model을 test window에서 최종 평가
7. permutation importance와 slice analysis로 실패를 해석

---

## 모델 비교

| 모델 | RMSE | MAE | R2 | FIT_SEC |
| --- | --- | --- | --- | --- |
| tuned_hist_gbdt | 60.0516 | 38.1593 | 0.9258 | - |
| poisson_baseline | 163.1814 | 120.1109 | 0.4522 | - |
| gpu_mlp | 164.3160 | 109.0213 | 0.4446 | - |

### 읽는 법

- tuned HGBDT가 평균 성능과 안정성 모두에서 우세했다.
- Poisson baseline은 구조가 단순해서 peak와 상호작용을 충분히 못 잡았다.
- GPU MLP는 학습은 되지만 tabular + 시간 패턴 문제에서는 tree boosting보다 덜 안정적이었다.

---

## CV 선택 근거

| 후보 | params | mean_rmse | std_rmse |
| --- | --- | --- | --- |
| candidate_1 | `learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, max_iter=160` | 67.6508 | 12.4027 |
| candidate_2 | `learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, max_iter=220` | 70.4737 | 15.1867 |
| candidate_3 | `learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=30, max_iter=280` | 74.1568 | 18.9147 |

핵심 메시지는 간단하다.

- 복잡도를 높인다고 항상 좋아지지 않는다.
- 시간-aware CV에서는 평균뿐 아니라 분산도 중요하다.
- candidate_1이 가장 낮은 평균 RMSE와 가장 작은 분산을 보여 줬다.

---

## 그림 읽기

### 결과 그림

- [CV fold RMSE boxplot](figures/results/cv_fold_score_boxplot.svg)
- [validation curve](figures/results/validation_curve.svg)
- [top feature importance](figures/results/top_feature_importance.svg)

### 해석 그림

- [subgroup metric comparison](figures/analysis/subgroup_metric_comparison.svg)
- [prediction-bin error plot](figures/analysis/confidence_bin_plot.svg)
- [common failure slice summary](figures/analysis/common_failure_slice_summary.svg)

---

## observed failure slice

아티팩트와 요약 파일을 함께 보면 실패는 다음 구간에 집중된다.

### workingday × weather 조합

| workingday | weather | MAE | 해석 |
| --- | --- | --- | --- |
| 0 | 3 | 73.30 | 가장 어려운 구간이다. 악천후 범주에서 오차가 크게 벌어진다. |
| 1 | 3 | 64.58 | 평일이어도 악천후면 오차가 여전히 크다. |
| 0 | 2 | 44.63 | 날씨가 조금만 나빠져도 오차가 증가한다. |
| 0 | 1 | 38.36 | 휴일/주말이라도 정상 날씨면 상대적으로 낫다. |
| 1 | 2 | 37.60 | 평일+보통 날씨는 중간 수준이다. |
| 1 | 1 | 32.04 | 가장 안정적인 쪽에 가깝다. |

### peak 시간대

worst-case 분석에서 오차는 다음 시간대에 몰렸다.

- 상위 시간대 분포: `{8: 9, 18: 4, 17: 4}`
- 상위 날씨 분포: `{3: 11, 1: 10, 2: 9}`
- 평균 실제 수요: `441.4`
- 평균 예측 수요: `401.6`

즉, 모델은 전체 평균은 괜찮게 맞추지만 **출퇴근 peak를 낮게 보는 보수적 예측**을 하는 경향이 있다.

### season 차이

`subgroup_metric_comparison.svg`에서는 season code별 RMSE도 달랐다.

- `season_1`: `67.114`
- `season_4`: `63.290`
- `season_3`: `51.424`

계절별 난이도 차이가 평균 점수 뒤에 숨어 있었다.

---

## feature importance 메모

`top_feature_importance.svg`의 상위 feature는 다음과 같다.

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

읽는 포인트는 하나다. **시간대 정보가 가장 강하고, 기상/캘린더 변수가 그 뒤를 따른다.**

---

## 최종 메모

이 단계에서 진짜로 배워야 하는 것은 “어떤 모델이 이겼는가”가 아니라, **왜 이 모델이 선택되었고, 어디서 틀렸으며, 다음에 무엇을 실험해야 하는가**를 설명하는 능력이다.

이번 Bike Sharing 실험은 그 구조를 잘 보여 준다. tuned HGBDT가 평균 성능과 안정성에서 가장 좋았고, 실패는 주로 악천후와 출퇴근 peak에 몰려 있었다. 다음 실험은 peak 대응과 slice 개선에 집중해야 한다.
