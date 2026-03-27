# 03. 모델 선택과 해석 실행 요약

- 과제: Bike Sharing 시간축 count 회귀
- 최고 모델: `tuned_hist_gbdt`
- 핵심 지표: RMSE=60.0516, MAE=38.1593, R2=0.9258

## 모델 비교

| 모델 | RMSE | MAE | R2 |
| --- | --- | --- | --- |
| tuned_hist_gbdt | 60.0516 | 38.1593 | 0.9258 |
| poisson_baseline | 163.1814 | 120.1109 | 0.4522 |
| gpu_mlp | 163.9343 | 108.1063 | 0.4472 |

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 실패 사례: `predictions/worst_predictions.csv`
