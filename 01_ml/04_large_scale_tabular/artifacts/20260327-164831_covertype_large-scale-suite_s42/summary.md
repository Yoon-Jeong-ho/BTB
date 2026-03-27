# 04. 대규모 표형 데이터 실행 요약

- 과제: Covertype 대규모 다중분류
- 최고 모델: `xgboost_gpu`
- 핵심 지표: Macro-F1=0.9192, Accuracy=0.9377, Macro-Recall=0.9070

## 모델 비교

| 모델 | Macro-F1 | Accuracy | Macro-Recall | Fit sec |
| --- | --- | --- | --- | --- |
| xgboost_gpu | 0.9192 | 0.9377 | 0.9070 | 8.26 |
| hist_gbdt | 0.7981 | 0.8365 | 0.7765 | 8.63 |
| gpu_mlp | 0.7350 | 0.8352 | 0.7007 | 46.84 |
| shallow_tree | 0.6394 | 0.7797 | 0.5806 | 3.36 |
| sgd_linear | 0.4578 | 0.7090 | 0.4455 | 2.86 |

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 예측 샘플: `predictions/test_predictions_sample.csv`
