# 02. 표형 회귀 실행 요약

- 과제: California Housing 회귀
- 최고 모델: `hist_gbdt`
- 핵심 지표: RMSE=0.4717, MAE=0.3179, R2=0.8302

## 모델 비교

| 모델 | RMSE | MAE | R2 | Fit sec |
| --- | --- | --- | --- | --- |
| hist_gbdt | 0.4717 | 0.3179 | 0.8302 | 1.33 |
| random_forest | 0.5146 | 0.3363 | 0.7979 | 0.68 |
| gpu_mlp | 0.5815 | 0.4044 | 0.7419 | 3.66 |
| ridge | 0.7329 | 0.5354 | 0.5901 | 0.04 |
| linear_regression | 0.7329 | 0.5354 | 0.5901 | 0.01 |
| dummy_mean | 1.1448 | 0.9031 | -0.0000 | 0.01 |

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 실패 사례: `predictions/worst_predictions.csv`
