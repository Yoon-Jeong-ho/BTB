# 04 Large Scale Tabular 이론 노트

이 문서는 **큰 표형 분류 문제를 왜 따로 공부해야 하는지**를 설명하기 위한 강의 노트다. 단순히 Covertype 점수를 외우기 위한 문서가 아니라, 대규모 tabular에서 어떤 질문이 생기고 왜 accuracy 하나로는 부족해지는지, 왜 macro 지표와 비용 지표를 같이 읽어야 하는지를 차근차근 정리한다.

- 단계 안내: [README.md](README.md)
- 최신 실험 리포트: [artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md](artifacts/20260327-164831_covertype_large-scale-suite_s42/README.md)
- 최신 요약: [artifacts/20260327-164831_covertype_large-scale-suite_s42/summary.md](artifacts/20260327-164831_covertype_large-scale-suite_s42/summary.md)

## 1. 왜 이런 이론이 필요해졌는가

작은 데이터에서는 모델 하나를 학습하고 점수표를 비교하는 것만으로도 꽤 많은 것을 배울 수 있다. 하지만 데이터가 커지면 상황이 달라진다.

- 실험 한 번에 드는 시간이 길어진다.
- 메모리와 디스크 사용량이 무시하기 어려워진다.
- class가 불균형하면 accuracy가 지나치게 낙관적으로 보인다.
- strong baseline을 돌리는 것 자체가 비용이 된다.

그래서 large-scale tabular에서는 질문이 바뀐다.

> “무슨 모델이 제일 높은 점수를 냈는가?”

에서

> “어떤 모델이 모든 class를 고르게 맞히면서도, 반복 실험이 가능한 비용 구조를 가지는가?”

로 바뀐다.

## 2. multiclass classification은 무엇을 푸는가

multiclass classification은 입력 feature를 보고 여러 class 중 하나를 고르는 문제다. 이 stage의 Covertype에서는 각 샘플이 어떤 산림 cover type에 속하는지를 맞힌다.

### binary classification과 다른 점

- decision boundary가 하나가 아니라 여러 class 쌍 사이에 생긴다.
- 어떤 class는 쉽게 구분되지만, 어떤 class는 서로 많이 섞인다.
- 전체 accuracy가 높아도 일부 class는 계속 잘못 예측될 수 있다.

그래서 multiclass 문제에서는 **전체 평균 + class별 세부 성능**을 같이 봐야 한다.

## 3. accuracy는 왜 쉽게 우리를 속이는가

accuracy는 전체 샘플 중 맞힌 비율이므로 직관적이다. 하지만 대규모 multiclass에서 가장 먼저 무너지는 지점도 accuracy다.

### 왜 그런가

1. 많이 등장하는 class를 잘 맞히면 점수가 쉽게 오른다.
2. 희소한 class가 무너져도 평균에 묻힌다.
3. 어떤 class pair가 서로 심하게 헷갈려도 accuracy만 보면 잘 드러나지 않는다.

Covertype 같은 데이터에서는 class 0/1이 많이 나오기 때문에, 이 둘을 잘 맞히는 모델은 accuracy가 꽤 높게 나온다. 그러나 class 4처럼 어려운 class가 계속 깨질 수 있다. 이 차이를 읽기 위해 macro metric이 필요하다.

## 4. macro-F1과 macro-recall은 무엇을 해결하는가

macro 계열 지표는 class마다 점수를 하나씩 계산한 뒤 평균낸다. 즉, class의 샘플 수가 많든 적든 **한 class로서 같은 발언권**을 준다.

### macro-recall
각 class의 정답을 얼마나 놓치지 않았는지 본다. class별 누락이 큰 문제에서 중요하다.

### macro-F1
precision과 recall 균형을 class마다 계산해서 평균낸다. minority class가 dominant class에 빨려 들어가는 현상을 더 민감하게 드러낸다.

### 왜 이 stage에서 중요한가

이번 실험의 최고 모델은 accuracy뿐 아니라 macro-F1과 macro-recall도 가장 높았다. 반대로 어떤 모델은 accuracy는 나쁘지 않지만 macro-F1이 많이 떨어졌다. 이 차이가 바로 “전체 평균은 괜찮아 보여도 일부 class를 제대로 못 잡는다”는 뜻이다.

## 5. cost-quality trade-off는 왜 연구와 운영을 동시에 생각하게 만드는가

실험이 비싸지면 모델 선택 기준도 달라진다. 같은 점수라면 더 가볍고 빠른 모델이 실전에서 유리하다. 조금 더 좋은 점수를 얻더라도 비용이 크게 늘면, 다음 실험을 못 돌릴 수 있다.

이 stage에서 기록하는 비용 지표는 아래와 같다.

- `fit time`: 학습 회전율
- `predict time`: 배포 후 응답성
- `peak RSS`: CPU 메모리 압박
- `peak GPU memory`: GPU 자원 압박

이 지표들이 필요한 이유는 모델을 **잘 맞히는 기계**가 아니라 **반복 가능한 실험 단위**로 보기 위해서다.

## 6. 왜 GPU boosting이 등장했는가

tree boosting은 tabular에서 강한 계열이지만, 데이터가 커질수록 split 탐색 비용이 커진다. GPU boosting은 histogram 기반 탐색과 병렬 연산을 이용해 이 비용을 줄이려는 접근이다.

### 기대하는 것

- strong baseline을 더 빠르게 학습
- 더 복잡한 비선형 경계를 효율적으로 학습
- 대형 데이터에서도 실험 회전율 확보

### 동시에 주의할 것

- 전처리가 CPU 쪽에 남아 있으면 병목이 생길 수 있다.
- GPU 사용량이 낮아도 전체 실험 시간이 빠르다는 뜻은 아니다.
- 품질 향상이 메모리 증가를 정당화하는지 같이 봐야 한다.

## 7. 이번 실험에 등장하는 모델들은 어디에 쓰이는가

### SGD Linear
매우 빠른 baseline이 필요할 때 쓴다. 선형 분리가 어느 정도 가능한지 빠르게 확인하는 용도다.

### Shallow Tree / Random Forest 계열
가벼운 비선형성을 빠르게 확인할 때 좋다. 복잡한 상호작용을 완전히 잡진 못해도, 선형 baseline보다 얼마나 나아지는지 보여 준다.

### HistGradientBoosting
scikit-learn 안에서 구현 가능한 strong baseline이다. tabular 문제에서 “기본적으로 이 정도는 나와야 한다”는 기준점이 된다.

### XGBoost GPU
품질과 속도를 동시에 노릴 수 있는 strong baseline이다. 대규모 tabular에서 가장 먼저 검토할 만한 후보 중 하나다.

### GPU MLP
GPU 신경망 비교군이다. 표현력은 좋지만, tabular에서 항상 tree 계열을 이기지 않는다는 점을 보여 주는 학습용 대조군이다.

## 8. 데이터셋을 어떻게 읽어야 하는가

Covertype를 읽을 때는 feature 이름보다 먼저 **class 구조와 크기**를 본다.

- class 분포가 균형적인가?
- 어떤 class가 가장 어렵나?
- 어떤 class pair가 자주 섞이나?
- 모델이 높은 confidence로도 특정 class를 자주 헷갈리나?

이런 질문은 단순 점수표보다 confusion matrix와 class-wise recall에서 더 잘 드러난다.

## 9. figure는 어떻게 읽는가

### `metric_vs_training_time.svg`
높은 macro-F1을 얻기 위해 얼마나 오래 학습해야 했는지 본다.

### `metric_vs_memory.svg`
점수를 조금 더 올리기 위해 메모리 비용이 얼마나 커졌는지 본다.

### `slice_metric_by_class.svg`
class별 recall 차이를 본다. 어떤 class가 약한지 바로 드러난다.

### `throughput_bottleneck_summary.svg`
학습/추론/메모리 중 어디가 병목인지 요약해서 본다.

### `sampling_strategy_performance.svg`
데이터 양이 늘어날 때 성능이 더 좋아질 여지가 있는지 본다.

## 10. 이번 stage에서 특히 주의 깊게 볼 것

1. accuracy와 macro-F1 사이 간격이 큰가?
2. class 0/1처럼 큰 class 쌍이 서로 많이 섞이는가?
3. class 4처럼 낮은 recall class가 계속 남는가?
4. 품질 향상이 fit time / memory 증가를 정당화하는가?
5. GPU 사용이 실제로 더 좋은 전체 실험 속도로 이어졌는가?

## 11. 다음 단계로 이어지는 질문

- class imbalance 완화를 위해 loss나 sampling을 어떻게 바꿔 볼 수 있을까?
- feature engineering을 넣으면 0/1 confusion이 줄어들까?
- 더 큰 데이터(HIGGS)로 올라갔을 때도 같은 strong baseline이 유지될까?
- 운영 제약이 더 강한 환경에서는 어떤 모델이 남을까?
