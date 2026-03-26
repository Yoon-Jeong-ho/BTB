# 04. 대규모 표형 데이터 결과 요약

## 한 줄 결론

- 과제: Covertype 대규모 다중분류
- 최고 모델: `xgboost_gpu`
- 핵심 지표: `macro_f1`=0.9192, `accuracy`=0.9377, `macro_recall`=0.9070, `mean_confidence`=0.8861
- 해석: 대규모 데이터에서는 비용 대비 성능 비교가 핵심이었고, class별 recall 편차를 함께 봐야 했다.

## 이론 포인트

- 상세 이론 문서: [04. 대규모 표형 데이터 THEORY](../../../../01_ml/04_large_scale_tabular/THEORY.md)

- 대규모 multiclass에서는 accuracy만 보면 class별 성능 붕괴를 놓칠 수 있으므로 macro-F1과 macro-recall을 같이 봐야 한다.
- throughput, fit time, peak memory는 서버 이전 전 단계에서 반드시 기록해야 하는 운영 지표다.
- GPU tree boosting은 대형 tabular strong baseline이지만, CPU/GPU 간 입력 위치 차이도 성능 해석에 포함해야 한다.

## 모델 비교

| 모델 | MACRO_F1 | ACCURACY | MACRO_RECALL | MEAN_CONFIDENCE | FIT_SEC |
| --- | --- | --- | --- | --- | --- |
| xgboost_gpu | 0.9192 | 0.9377 | 0.9070 | 0.8861 | 7.94 |
| hist_gbdt | 0.7981 | 0.8365 | 0.7765 | 0.7849 | 8.22 |
| gpu_mlp | 0.7369 | 0.8383 | 0.7046 | 0.8101 | 29.13 |
| shallow_tree | 0.6394 | 0.7797 | 0.5806 | 0.6667 | 3.36 |
| sgd_linear | 0.4578 | 0.7090 | 0.4455 | 0.6960 | 2.62 |

## 결과 해석 / 실패 분석

- class별 recall은 최저 4번=0.802, 최고 6번=0.958 로 편차가 분명했다.
- 가장 큰 confusion pair 상위 3개는 [((0, 1), 2562), ((1, 0), 1573), ((4, 1), 257)] 였다.
- 즉, 전체 accuracy가 높아도 특정 cover type 쌍은 여전히 자주 헷갈리므로 macro 지표를 계속 추적해야 한다.

## 결과 Figure

### metric_vs_training_time.svg

![](figures/results/metric_vs_training_time.svg)

### metric_vs_memory.svg

![](figures/results/metric_vs_memory.svg)

### score_distribution.svg

![](figures/results/score_distribution.svg)


## 분석 Figure

### slice_metric_by_class.svg

![](figures/analysis/slice_metric_by_class.svg)

### throughput_bottleneck_summary.svg

![](figures/analysis/throughput_bottleneck_summary.svg)

### sampling_strategy_performance.svg

![](figures/analysis/sampling_strategy_performance.svg)


## 다음 액션

- 최고 점수만 보지 말고, figure에서 드러난 실패 slice를 다음 실험 가설로 연결한다.
- 원시 산출물은 `runs/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/` 아래에서 확인할 수 있다.
