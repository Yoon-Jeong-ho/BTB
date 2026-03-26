# 02. 표형 회귀 결과 요약

## 한 줄 결론

- 과제: California Housing 회귀
- 최고 모델: `hist_gbdt`
- 핵심 지표: `rmse`=0.4717, `mae`=0.3179, `r2`=0.8302
- 해석: HistGradientBoostingRegressor가 가장 낮은 RMSE를 기록했고, 고가 주택/특정 지역에서 residual이 커졌다.

## 이론 포인트

- 상세 이론 문서: [02. 표형 회귀 THEORY](../../../../01_ml/02_tabular_regression/THEORY.md)

- RMSE는 큰 오차에 민감하고, MAE는 평균적인 오차 크기를 직관적으로 보여 준다.
- parity plot과 residual plot을 함께 봐야 특정 target 구간에서 systematic bias가 있는지 알 수 있다.
- tabular 회귀에서는 선형 모델과 tree 모델의 bias 패턴 차이를 비교하는 것이 중요하다.

## 모델 비교

| 모델 | RMSE | MAE | R2 | FIT_SEC |
| --- | --- | --- | --- | --- |
| hist_gbdt | 0.4717 | 0.3179 | 0.8302 | 1.56 |
| random_forest | 0.5146 | 0.3363 | 0.7979 | 0.52 |
| gpu_mlp | 0.5848 | 0.4032 | 0.7391 | 1.92 |
| ridge | 0.7329 | 0.5354 | 0.5901 | 0.04 |
| linear_regression | 0.7329 | 0.5354 | 0.5901 | 0.01 |
| dummy_mean | 1.1448 | 0.9031 | -0.0000 | 0.01 |

## 결과 해석 / 실패 분석

- 최악 예측 30개에서 평균 실제값은 3.616, 평균 예측값은 2.489 으로 전반적으로 과소추정 경향이 있었다.
- 최악 사례 중 고가 주택(`target>4.5`) 비중은 43.3% 로, 상단 tail을 충분히 따라가지 못했다.
- 반대로 저가 구간(`target<1.0`) 비중도 13.3% 존재해, 양끝단에서 residual이 커지는 구조가 확인됐다.

## 결과 Figure

### parity_plot.svg

![](figures/results/parity_plot.svg)

### residual_vs_target.svg

![](figures/results/residual_vs_target.svg)

### learning_curve.svg

![](figures/results/learning_curve.svg)


## 분석 Figure

### feature_importance.svg

![](figures/analysis/feature_importance.svg)

### regional_error_slice.svg

![](figures/analysis/regional_error_slice.svg)

### worst_prediction_cases.svg

![](figures/analysis/worst_prediction_cases.svg)


## 다음 액션

- 최고 점수만 보지 말고, figure에서 드러난 실패 slice를 다음 실험 가설로 연결한다.
- 원시 산출물은 `runs/01_ml/02_tabular_regression/20260326-172452_california-housing_model-suite_s42/` 아래에서 확인할 수 있다.
