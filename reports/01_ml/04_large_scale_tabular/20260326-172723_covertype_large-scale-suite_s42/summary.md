# 04. 대규모 표형 데이터 결과 요약

## 핵심 결론

- 과제: Covertype 대규모 다중분류
- 최고 모델: `xgboost_gpu`
- 핵심 지표: `macro_f1`=0.9192, `accuracy`=0.9377, `macro_recall`=0.9070, `mean_confidence`=0.8861
- 요약: accuracy는 높았지만, class별 recall과 confusion pattern을 함께 보지 않으면 중요한 실패를 놓칠 수 있었다.

## 이론과 연결

- 이론 문서: [01_ml/04_large_scale_tabular/THEORY.md](../../../../01_ml/04_large_scale_tabular/THEORY.md)
- 불균형 multiclass에서는 accuracy보다 macro-F1과 macro-recall이 더 잘 문제를 드러낸다.
- throughput, predict time, peak memory는 대규모 tabular에서 반드시 함께 기록해야 하는 운영 지표다.
- GPU boosting은 강력하지만, 데이터 이동과 메모리 사용까지 포함해 해석해야 한다.

## 왜 이 이론이 필요했나

### large-scale tabular

이 데이터는 단순한 표형 데이터가 아니라, **많은 샘플을 빠르게 반복 실험해야 하는 표형 데이터**다.
여기서는 성능이 좋아도 학습이 느리거나 메모리가 크면 실험 루프가 깨진다.
그래서 이론 문서는 품질만이 아니라 **속도와 메모리까지 같이 읽어야 한다**고 강조한다.

### macro metrics

artifact를 보면 class 0/1이 많이 등장하고 class 3은 희소하다.
이런 상황에서는 accuracy가 minority class의 실패를 가릴 수 있다.
그래서 macro-F1과 macro-recall이 필요하다.
즉, macro 지표는 **다수 class 편향을 누르고 class별 실패를 드러내는 도구**다.

### cost-quality trade-off

대규모 tabular에서는 점수만 높은 모델보다 **점수 대비 비용이 좋은 모델**이 더 중요하다.
이 stage는 fit time, predict time, peak memory를 같이 기록해서
- 더 좋은 점수를 얻는 데 드는 비용이 무엇인지
- 그 비용이 감당 가능한지
를 함께 본다.

### GPU boosting

tree boosting은 tabular에서 강력한 계열이지만, 큰 데이터에서는 CPU만으로 실험 속도가 부족할 수 있다.
GPU boosting은 histogram 기반 split 탐색과 병렬 연산을 이용해 이 문제를 해결하려고 등장했다.
이 stage에서는 GPU를 썼다는 사실보다, **강한 tree 계열을 더 좋은 비용 구조로 돌릴 수 있는가**가 중요했다.

## 실험 해석

### 왜 xgboost_gpu가 가장 좋았나

- macro-F1 `0.9192`는 다른 모델보다 한참 높다.
- accuracy `0.9377`과 macro-recall `0.9070`도 가장 좋다.
- fit time `7.94s`는 hist_gbdt와 비슷하지만, 품질은 훨씬 높았다.
- predict time `0.10s`로 추론도 빠르다.

즉, xgboost_gpu는 이 데이터에서 **품질과 속도의 균형이 가장 좋았던 모델**이다.

### hist_gbdt와 비교하면

- hist_gbdt는 strong baseline이 맞지만, macro-F1 `0.7981`에 머물렀다.
- peak RSS는 `4375MB` 수준으로 낮은 편이지만, 성능 격차가 꽤 크다.
- 따라서 "더 가볍다"는 장점이 있어도, 현재 문제에서는 xgboost_gpu만큼의 정밀한 경계를 만들지 못했다.

### gpu_mlp와 비교하면

- gpu_mlp는 fit time `29.13s`로 가장 느렸다.
- macro-F1 `0.7369`, macro-recall `0.7046`으로 tree 계열보다 약했다.
- peak GPU memory는 `207.2MB`였지만, 이 숫자가 곧 더 좋은 모델을 뜻하지는 않는다.

즉, tabular에서는 GPU를 쓰는 것보다 **문제 구조에 맞는 모델 계열을 고르는 것**이 더 중요하다.

## 실패 분석

### class별 recall

- 최저 recall class: `4번 = 0.8020`
- 최고 recall class: `6번 = 0.9584`

### 큰 confusion pair

- `(0 -> 1)`: `2562`
- `(1 -> 0)`: `1573`
- `(4 -> 1)`: `257`
- `(5 -> 2)`: `171`
- `(2 -> 5)`: `121`

이 패턴은 dominant class 주변에서 boundary가 흔들리고, minority class가 dominant class로 흡수되는 경향을 보여 준다.

## figure 읽는 법

### 결과 Figure

- `metric_vs_training_time.svg`: 점수와 학습 시간을 함께 보면 어떤 모델이 실험 효율이 좋은지 알 수 있다.
- `metric_vs_memory.svg`: 높은 점수가 더 큰 메모리 비용을 요구하는지 확인할 수 있다.
- `score_distribution.svg`: 예측 confidence가 극단적으로 몰려 있는지 볼 수 있다.

### 분석 Figure

- `slice_metric_by_class.svg`: class별 recall 차이를 본다.
- `throughput_bottleneck_summary.svg`: 모델별 시간/메모리 병목을 정리한다.
- `sampling_strategy_performance.svg`: 더 많은 데이터가 들어오면 성능이 계속 오르는지 본다.

## 다음 가설

1. class 0/1 confusion을 줄이기 위한 더 세밀한 feature 분석
2. class 4 recall 개선을 위한 class-balanced 학습 전략
3. xgboost_gpu 설정 최적화로 메모리 비용을 더 줄일 수 있는지 확인
4. HIGGS 같은 더 큰 데이터로 문서 구조와 실험 규약을 확장

## 관련 링크

- 최신 리포트: [README.md](README.md)
- 최신 결과 요약: [summary.md](summary.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
