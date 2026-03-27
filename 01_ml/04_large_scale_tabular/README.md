# 04 Large Scale Tabular

이 단계는 대규모 표형 데이터를 다룰 때 **정확도, 균형성, 속도, 메모리**를 함께 읽는 법을 익히는 자리다.
이번 문서는 단순한 개념 소개가 아니라, **왜 이론이 등장했는지**, **무엇을 해결하려는지**, **실험 결과를 어떻게 해석해야 하는지**를 연결해서 읽도록 만든 공부노트다.

## 이 단계에서 먼저 잡아야 할 질문

| 질문 | 이 질문이 생긴 이유 | 이 단계에서 얻어야 할 답 |
| --- | --- | --- |
| 왜 accuracy만 보면 안 되는가? | class가 불균형하면 다수 class가 점수를 끌어올리기 때문 | macro-F1 / macro-recall을 같이 봐야 한다 |
| 왜 macro metric이 필요한가? | minority class 실패가 전체 점수에 묻히기 때문 | class별 성능을 동등한 비중으로 읽어야 한다 |
| 왜 cost-quality trade-off를 보나? | 점수만 좋고 너무 느리거나 무거우면 실전에서 못 쓰기 때문 | 품질과 비용을 함께 비교해야 한다 |
| 왜 GPU boosting이 중요한가? | tabular의 strong baseline을 더 빠르게 돌리고 싶기 때문 | 대규모 데이터에서 tree boosting을 효율적으로 학습할 수 있다 |
| 왜 confusion matrix를 읽나? | 어떤 class 경계가 무너졌는지 알아야 다음 실험을 설계할 수 있기 때문 | class pair별 혼동을 가설로 바꿀 수 있다 |

## 먼저 읽을 문서

1. [이론 문서](THEORY.md)
2. [최신 리포트 README](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/README.md)
3. [리포트 요약](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/summary.md)

## 이번 단계의 핵심 개념

### 1) large-scale tabular

- 샘플 수가 크면 학습 시간이 곧 실험 회전율이 된다.
- 모델이 조금만 무거워져도 메모리와 추론 지연이 운영 이슈가 된다.
- 그래서 이 단계는 "점수가 높은 모델"보다 **반복 가능하고 비교 가능한 모델**을 찾는 연습이다.

### 2) macro metrics

- accuracy는 전체 평균만 보여 준다.
- macro-F1 / macro-recall은 class를 동등하게 취급한다.
- 불균형 multiclass에서는 이 차이가 매우 중요하다.

### 3) cost-quality trade-off

- 성능이 같다면 더 빠르고 가벼운 모델이 좋다.
- 성능이 조금 좋아져도 비용이 크게 늘면 다시 생각해야 한다.
- 이 관점이 있어야 연구용 최고 점수와 실전용 최적점을 구분할 수 있다.

### 4) GPU boosting

- histogram 기반 split 탐색을 GPU 병렬 처리로 가속한다.
- tabular에서 강한 tree boosting 계열을 더 빠르게 학습하려는 시도다.
- 하지만 데이터 이동, 전처리, 메모리 배치까지 같이 봐야 한다.

## 이번 프로젝트 기준 확정 데이터셋

- 실행 코드: `run_stage.py`
- 이론 문서: [THEORY.md](THEORY.md)
- 최신 report: [`reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/README.md`](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/README.md)
- report summary: [`reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/summary.md`](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/summary.md)

- Primary: `mstz/covertype`
- Source: Hugging Face Datasets
- Load:

```python
from datasets import load_dataset

ds = load_dataset("mstz/covertype", split="train")
df = ds.to_pandas()
```

- 이유: Hugging Face로 바로 접근 가능하고, 로컬에서 다루기엔 크고 서버 실습용으론 아직 관리 가능한 수준이다.
- Server extension: `HIGGS`

## 실험 구조를 읽는 법

### 1. 이론을 먼저 고정한다

이 stage에서 읽어야 할 핵심 이론은 다음과 같다.

- accuracy와 macro metric의 차이
- 왜 macro-recall이 필요한가
- 왜 cost-quality trade-off를 봐야 하는가
- 왜 GPU boosting이 강한가
- confusion matrix와 class별 recall을 어떻게 연결하는가

### 2. 실험을 그 위에 올린다

비교한 모델은 다음과 같다.

- `sgd_linear`: 가장 가벼운 약한 baseline
- `shallow_tree`: 얕은 tree 기반 baseline
- `hist_gbdt`: scikit-learn strong baseline
- `xgboost_gpu`: GPU boosting strong baseline
- `gpu_mlp`: neural network 대조군

### 3. 결과를 해석한다

좋은 결과는 단순히 점수가 높은 것이 아니라, 아래를 함께 만족해야 한다.

- macro 지표가 좋아진다
- class별 recall 편차가 줄어든다
- 학습 시간이 과도하게 늘지 않는다
- 메모리 비용이 감당 가능하다
- confusion pair가 줄어든다

## 실습 파이프라인

1. 데이터 원본은 Git 밖에 두고 fetch/prep 스크립트만 버전 관리한다.
2. schema와 split manifest를 고정한다.
3. baseline으로 선형/얕은 tree 모델부터 시작한다.
4. histogram 기반 GBDT 또는 boosted tree로 확장한다.
5. metric뿐 아니라 학습 시간, peak memory, 데이터 적재 시간을 같이 기록한다.
6. checkpoint와 중간 feature matrix는 외부 저장소로 분리한다.

## 결과로 남길 figure

아래 figure는 "정확도가 얼마나 나왔는가"를 보여 주는 결과 시각화다.

- [metric_vs_training_time.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/metric_vs_training_time.svg)
- [metric_vs_memory.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/metric_vs_memory.svg)
- [score_distribution.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/results/score_distribution.svg)

## 분석으로 남길 figure

아래 figure는 "왜 그런 결과가 나왔는가"를 읽는 분석 시각화다.

- [slice_metric_by_class.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/slice_metric_by_class.svg)
- [throughput_bottleneck_summary.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/throughput_bottleneck_summary.svg)
- [sampling_strategy_performance.svg](../../reports/01_ml/04_large_scale_tabular/20260326-172723_covertype_large-scale-suite_s42/figures/analysis/sampling_strategy_performance.svg)

## 결과를 읽을 때 기억할 포인트

- accuracy가 높아도 macro-F1이 낮을 수 있다.
- class별 recall이 낮으면, 특정 class boundary가 무너졌다는 뜻이다.
- fit time과 peak memory는 실험 속도와 운영 비용을 결정한다.
- GPU boosting은 대규모 tabular에서 강력하지만, 데이터 이동과 메모리 배치를 함께 봐야 한다.
- confusion matrix는 다음 실험의 가설을 만드는 출발점이다.

## 승격 기준

- 같은 규약으로 로컬 실험과 서버 실험이 비교 가능하다.
- 어떤 artifact를 Git에 두고 어떤 artifact를 외부에 둬야 하는지 명확하다.
- report를 읽었을 때 "무엇이 좋았는지"보다 "왜 그렇게 나왔는지"를 설명할 수 있어야 한다.
