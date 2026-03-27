# 04. 대규모 표형 데이터 결과 요약

## 핵심 결론

- 과제: Covertype 대규모 다중분류
- 최고 모델: `xgboost_gpu`
- 핵심 지표: `macro_f1`=0.9192, `accuracy`=0.9377, `macro_recall`=0.9070, `mean_confidence`=0.8861
- 한 줄 해석: 이 실험의 핵심은 "정확도가 높다"가 아니라, **macro 지표와 비용 지표를 함께 보니 `xgboost_gpu`가 가장 균형이 좋았다**는 점이다.

## 이론 연결

- [이 stage 소개](../../README.md)
- [이론 노트](../../THEORY.md)
- accuracy는 다수 class가 점수를 끌어올릴 수 있다.
- macro-F1 / macro-recall은 class별 실패를 드러낸다.
- fit/predict time과 peak RSS는 large-scale 실험에서 반드시 함께 봐야 한다.

## 실험 해석

- `xgboost_gpu`는 품질과 속도의 균형이 가장 좋았다.
- `hist_gbdt`는 strong baseline으로는 충분히 좋지만 class-wise 성능 차이가 더 컸다.
- `gpu_mlp`는 GPU를 써도 tabular strong baseline을 이기지 못할 수 있다는 점을 보여 줬다.

## 실패 포인트

- 최저 recall class: `4번 = 0.8020`
- 최고 recall class: `6번 = 0.9584`
- 큰 confusion pair: `(0→1)`, `(1→0)`, `(4→1)`

## Figure 바로가기

### 결과 Figure
- [metric_vs_training_time.svg](figures/results/metric_vs_training_time.svg)
- [metric_vs_memory.svg](figures/results/metric_vs_memory.svg)
- [score_distribution.svg](figures/results/score_distribution.svg)

### 분석 Figure
- [slice_metric_by_class.svg](figures/analysis/slice_metric_by_class.svg)
- [throughput_bottleneck_summary.svg](figures/analysis/throughput_bottleneck_summary.svg)
- [sampling_strategy_performance.svg](figures/analysis/sampling_strategy_performance.svg)
