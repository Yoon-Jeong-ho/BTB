# 04 Large Scale Tabular 이론

이 장은 **큰 표형 데이터에서 왜 accuracy만으로는 부족한지**, **왜 macro 지표와 비용 지표를 함께 읽어야 하는지**, **왜 GPU boosting이 실전에서 자주 강한지**를 공부노트 형태로 정리한 문서다.

## 한 줄 요약

대규모 tabular에서는 "정확히 맞히는가"만이 아니라 "모든 class를 고르게 맞히는가", "그 성능을 얼마나 빨리/가볍게 얻는가"까지 같이 봐야 한다.

## 1. 왜 이런 이론이 등장했나

대규모 tabular는 단순히 샘플 수가 많은 문제가 아니다.
문제는 **실험이 비싸고, class가 불균형하고, 운영 제약이 강하다는 점**이다.

| 상황 | 생기는 문제 | 기존 관점의 한계 | 이 장이 해결하려는 것 |
| --- | --- | --- | --- |
| 데이터가 큼 | 학습/검증 반복이 느려짐 | 점수만 보고 모델을 고르기 어려움 | 실험 회전율과 재현성 관리 |
| class 불균형 | 다수 class가 점수를 끌어올림 | accuracy가 minority 실패를 가림 | macro-F1 / macro-recall 사용 |
| 모델이 큼 | 메모리와 추론 지연이 커짐 | 점수만으로 운영 가능성 판단 불가 | cost-quality trade-off 읽기 |
| GPU 사용 | 빠를 것 같지만 이동 비용이 있음 | "GPU = 항상 빠름"이라는 오해 | GPU boosting의 실제 병목 이해 |

즉, 이 장의 출발점은 "가장 좋은 모델"이 아니라 **실험 가능한 모델**과 **운영 가능한 모델**을 구분하는 데 있다.

## 2. large-scale tabular은 무엇을 다르게 보게 만드는가

대규모 표형 데이터에서는 다음 네 축을 동시에 본다.

1. **품질**: accuracy, macro-F1, macro-recall
2. **속도**: fit time, predict time
3. **메모리**: peak RSS, peak GPU memory
4. **안정성**: class별 recall, confusion pattern

작은 데이터에서는 품질만 높으면 좋은 모델처럼 보일 수 있다.
하지만 대규모 tabular에서는 품질이 비슷해도 학습이 너무 느리거나 메모리가 너무 크면 실험 루프가 무너진다.
그래서 이 장은 **성능 지표와 시스템 지표를 같이 읽는 습관**을 기르는 데 목적이 있다.

## 3. accuracy만으로는 왜 부족한가

accuracy는 전체 샘플 중 맞힌 비율이다.

```text
accuracy = 맞힌 샘플 수 / 전체 샘플 수
```

이 정의는 단순하지만, multiclass + imbalance 상황에서는 위험하다.
다수 class를 잘 맞히기만 해도 accuracy가 높아질 수 있기 때문이다.

### 이번 stage에서 accuracy가 놓치는 것

- class 0과 class 1이 많이 등장하므로, 이 둘만 잘 맞혀도 전체 accuracy는 쉽게 좋아진다.
- 반대로 class 3, class 4처럼 상대적으로 적거나 경계가 애매한 class는 쉽게 묻힌다.
- 따라서 accuracy가 높더라도 minority class의 실패가 계속 남을 수 있다.

즉, accuracy는 "전체 평균"은 보여 주지만, **어떤 class가 망가졌는지**는 잘 말해 주지 않는다.

## 4. macro metrics는 무엇을 해결하는가

macro 지표는 각 class를 똑같은 비중으로 평균낸다.

```text
macro-recall = (1 / C) * Σ recall_c
macro-F1     = (1 / C) * Σ F1_c
```

여기서 `C`는 class 수다.

### macro 지표의 역할

- **majority class 편향 완화**: 많이 나온 class가 점수를 독점하지 못하게 한다.
- **minority class 보호**: 샘플 수가 적어도 한 class로서의 실패가 드러난다.
- **경계 붕괴 탐지**: 특정 class pair가 자꾸 섞이면 macro recall/F1이 바로 떨어진다.

### 왜 이번 실험에서 특히 중요했나

Covertype는 class 분포가 고르지 않다.
artifact에서도 class 0과 1이 압도적으로 많고, class 3은 희소하다는 점이 보인다.
이런 데이터에서는 accuracy만 보면 "괜찮아 보이는" 모델이 실제로는 몇몇 class를 계속 놓칠 수 있다.

그래서 이 장에서는 **macro-F1과 macro-recall을 accuracy와 항상 같이 읽는다.**

## 5. cost-quality trade-off는 왜 중요한가

좋은 모델은 점수만 높은 모델이 아니다.
대규모 tabular에서는 **성능을 얼마나 싼 비용으로 얻었는가**가 바로 실험 가치가 된다.

### 이 trade-off가 해결하는 문제

- 같은 점수라도 학습이 10배 느리면 반복 실험이 어렵다.
- 같은 점수라도 메모리가 2배 크면 더 큰 데이터나 더 많은 실험을 동시에 못 돌린다.
- 추론이 느리면 운영 환경에서 배포가 어려워진다.

### 이 stage에서 보는 비용 지표

- `fit time`: 학습 회전율
- `predict time`: 운영 지연
- `peak RSS`: CPU 메모리 압박
- `peak GPU memory`: GPU 자원 압박

### 해석 원칙

- **품질이 조금 좋아졌는데 비용이 크게 늘면** trade-off가 나빠진다.
- **품질이 더 좋고 비용도 비슷하면** 그 모델이 우위다.
- **품질이 비슷하면 더 가볍고 빠른 모델이 실전에서 유리**하다.

이 관점이 중요한 이유는, 연구용 최고 점수와 실전용 최적점이 자주 다르기 때문이다.

## 6. GPU boosting은 왜 등장했나

Tree boosting은 tabular에서 오랫동안 강력한 계열이었다.
그런데 대규모 데이터에서는 **split 탐색과 histogram 생성 비용**이 커져 CPU만으로는 실험이 느려질 수 있다.

GPU boosting은 이 문제를 다음 방식으로 푼다.

1. 연속형 feature를 histogram/bin으로 압축한다.
2. split 후보 탐색을 병렬화한다.
3. GPU의 대규모 병렬 연산과 메모리 대역폭을 활용한다.

### 이 이론이 해결하는 것

- **대규모 tabular에서도 strong baseline을 빠르게 학습**
- **반복 실험 시간을 줄여 실험 회전율 개선**
- **강한 비선형 경계를 유지하면서도 더 큰 데이터 처리 가능**

### 하지만 자동으로 이기는 것은 아니다

GPU boosting은 강력하지만 다음 병목은 여전히 남는다.

- 데이터 이동 비용
- 전처리가 CPU에 남아 있는 문제
- GPU/CPU 메모리 배치
- 너무 큰 모델에서의 비용 증가

즉, GPU boosting은 "하드웨어만 바꾼 것"이 아니라 **알고리즘 + 시스템을 함께 설계하는 방식**이다.

## 7. 이번 stage에서 왜 HistGBDT와 XGBoost GPU를 같이 봤나

이 둘은 같은 tree boosting 계열이지만, 대규모 tabular에서 보는 질문이 다르다.

- `HistGradientBoostingClassifier`: scikit-learn 기준의 강한 baseline
- `XGBoost GPU`: 더 강한 최적화와 GPU 가속을 기대할 수 있는 baseline

둘을 같이 보면 다음을 알 수 있다.

- tree boosting 자체가 이 데이터에서 잘 맞는가
- histogram 기반 split이 충분한가
- GPU 최적화가 실제로 품질과 속도에 이득을 주는가

즉, 이 비교는 **"GPU를 썼는가"**가 아니라 **"같은 문제를 더 좋은 비용으로 푸는가"**를 묻는다.

## 8. confusion matrix와 class recall은 왜 같이 봐야 하나

confusion matrix는 단순한 오답표가 아니다.
**어떤 class가 어떤 class로 흡수되는지**를 보여 주는 경계 지도다.

class recall은 다음을 묻는다.

- "이 class의 정답을 얼마나 놓치지 않았는가?"

두 지표를 같이 보면 다음을 읽을 수 있다.

- recall이 낮은 class는 경계가 불안정하다.
- 특정 pair의 혼동이 크면 feature 공간에서 두 class가 가까이 붙어 있다.
- minority class가 dominant class로 빨려 들어가면 macro 지표가 먼저 반응한다.

### 이번 artifact에서 보인 패턴

- class 0 ↔ class 1 혼동이 가장 크다.
- class 4는 가장 낮은 recall을 기록했다.
- class 6은 가장 높은 recall을 보였다.

이 조합은 **전체 accuracy가 좋아도 class boundary는 아직 균일하지 않다**는 뜻이다.

## 9. 이번 stage 실험을 읽는 순서

1. class 분포를 본다.
2. baseline 모델과 strong baseline을 나눈다.
3. accuracy만 보지 말고 macro-F1 / macro-recall을 같이 본다.
4. fit time / predict time / peak memory를 같이 본다.
5. confusion pair와 class recall을 읽는다.
6. 다음 feature engineering 또는 sampling 가설을 세운다.

## 10. 꼭 기억할 문장

- accuracy는 "전체 평균"이고, macro 지표는 "class별 공정성"에 가깝다.
- cost-quality trade-off는 모델의 실전 적합성을 판단하는 핵심이다.
- GPU boosting은 대규모 tabular에서 강력하지만, 데이터 이동과 메모리까지 포함해 봐야 한다.
- confusion matrix는 실패를 보여 주는 도구이자 다음 실험의 출발점이다.

## 11. 연결 문서

- 단계 개요: [README.md](README.md)
- 최신 실험 리포트: [reports/.../README.md](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/README.md)
- 리포트 요약: [reports/.../summary.md](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/summary.md)
- 결과 figure:
  - [metric_vs_training_time.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/metric_vs_training_time.svg)
  - [metric_vs_memory.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/metric_vs_memory.svg)
  - [score_distribution.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/score_distribution.svg)
- 분석 figure:
  - [slice_metric_by_class.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/slice_metric_by_class.svg)
  - [throughput_bottleneck_summary.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/throughput_bottleneck_summary.svg)
  - [sampling_strategy_performance.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/sampling_strategy_performance.svg)
