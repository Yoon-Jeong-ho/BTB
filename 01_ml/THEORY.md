# 01 ML 이론 개요

이 문서는 `01_ml` 전체를 공부할 때 필요한 공통 이론 지도를 제공한다.
핵심 목표는 단순히 모델을 "돌릴 줄 아는 상태"가 아니라, **왜 이런 방법이 필요한지 설명하고, 결과를 해석하고, 다음 실험을 설계할 수 있는 상태**까지 가는 것이다.

이 트랙에서는 아래 흐름을 반복해서 익힌다.

1. 문제를 정의한다.
2. 데이터 split과 preprocessing을 설계한다.
3. baseline을 먼저 세운다.
4. strong baseline을 추가한다.
5. metric과 figure로 결과를 읽는다.
6. failure case를 보고 다음 가설을 만든다.

---

## 1. 왜 고전 ML부터 시작하는가

많은 사람이 딥러닝부터 시작하지만, 실제로는 ML 기본기가 약하면 이후 NLP/멀티모달에서도 같은 실수를 반복한다.

예를 들어:

- train/test leakage를 막지 못한다.
- accuracy만 보고 성능이 좋다고 착각한다.
- 전처리를 train/test에 따로 하지 않고 전체 데이터에 fit해 버린다.
- figure는 만들지만 해석은 하지 못한다.
- baseline 없이 강한 모델만 돌리고 만족한다.

그래서 `01_ml` 단계는 단순한 입문이 아니라, 이후 모든 실험의 습관을 만드는 단계다.

---

## 2. supervised learning은 어떤 문제를 푸는가

ML 트랙의 네 단계는 모두 supervised learning 문제다.

- 입력 `X`: feature
- 출력 `y`: label 또는 target
- 목표: `X -> y` 관계를 일반화해서 보지 못한 데이터에도 잘 맞추기

여기서 중요한 것은 "훈련 데이터에 맞추기"가 아니라 **일반화(generalization)** 다.

### 왜 일반화가 중요한가

훈련 데이터에서만 잘 맞는 모델은 암기한 것에 가깝다.
우리가 원하는 것은 새로운 샘플에서도 안정적으로 맞히는 모델이다.

그래서 항상 아래 질문을 해야 한다.

- 이 모델이 train에서만 잘 맞는가?
- valid/test에서도 유지되는가?
- 특정 slice에서만 잘 맞는 것은 아닌가?

---

## 3. 데이터 분할 이론

### 3-1. train / valid / test는 왜 나뉘는가

데이터를 나누는 이유는 역할이 다르기 때문이다.

- `train`: 모델 파라미터를 학습하는 데이터
- `valid`: 모델 선택, threshold 조정, 하이퍼파라미터 선택용 데이터
- `test`: 마지막에 단 한 번, 최종 성능을 확인하는 데이터

### 3-2. 왜 test를 자주 보면 안 되는가

test 성능을 보면서 반복적으로 의사결정을 하면, 결국 test에 맞춘 모델을 고르게 된다.
이것이 test leakage다.

겉으로는 test set을 안 건드린 것처럼 보여도,
실제로는 test 점수에 맞춰 실험 방향을 바꾸고 있으므로 일반화 추정이 오염된다.

### 3-3. 시간 데이터는 왜 split이 달라야 하는가

`Bike Sharing` 같은 시간 데이터는 미래가 과거보다 뒤에 있다.
random split을 쓰면 미래 정보가 과거 훈련에 섞인다.
이것은 실제 배포 상황과 다르다.

그래서 시간 데이터는:

- 미래를 test로 두고
- 과거에서만 학습하고
- fold도 시간 순서를 지키는 `TimeSeriesSplit`을 사용해야 한다.

즉, split 전략은 단순한 구현 디테일이 아니라 **문제 정의 자체의 일부**다.

---

## 4. 전처리 이론

### 4-1. 전처리는 왜 필요한가

표형 데이터는 대부분 아래 문제를 가진다.

- 결측치가 있다.
- 수치형/범주형이 섞여 있다.
- scale이 다르다.
- 같은 의미라도 표현 방식이 다르다.

모델은 이런 데이터를 그대로 잘 처리하지 못하는 경우가 많다.
그래서 전처리를 통해 모델이 읽기 쉬운 표현으로 바꿔 준다.

### 4-2. 결측치 처리

결측치는 단순히 빈칸이 아니라 정보 손실이다.
하지만 많은 모델은 NaN을 그대로 받지 못한다.

대표적인 방법:

- 수치형: median / mean 대체
- 범주형: 최빈값 대체
- 결측 자체를 별도 정보로 다루기

여기서 중요한 것은 "무조건 평균으로 메우기"가 아니라,
**왜 그 대체 방식이 현재 데이터에 적절한지** 설명할 수 있어야 한다는 점이다.

### 4-3. 범주형 인코딩

범주형 변수는 문자열 그대로는 모델이 이해하지 못한다.
그래서 one-hot encoding 같은 방식으로 바꾼다.

왜 one-hot을 쓰는가?

- 범주 사이에 임의의 순서를 넣지 않기 위해서다.
- 예를 들어 `Private=0`, `Self-emp=1`, `Gov=2` 로 두면,
  모델이 잘못하면 `Gov > Self-emp > Private` 같은 허구의 순서를 학습할 수 있다.

### 4-4. 스케일링

선형 모델이나 신경망은 feature scale에 민감할 수 있다.
반면 tree 모델은 상대적으로 덜 민감하다.

즉, preprocessing은 모델 종류와 연결해서 생각해야 한다.

### 4-5. 왜 파이프라인이 중요한가

전처리를 수동으로 하면 leakage가 쉽게 생긴다.
예를 들어 전체 데이터의 평균으로 결측치를 채우면 test 정보가 train에 들어간다.

`Pipeline`, `ColumnTransformer`를 쓰는 이유는 단순히 편해서가 아니라,
**fit은 train에만, transform은 valid/test에만 적용되도록 구조적으로 강제하기 위해서**다.

---

## 5. baseline 이론

### 5-1. baseline은 왜 반드시 필요한가

강한 모델 하나만 돌리면 점수가 좋아도 의미를 알 수 없다.
비교 기준이 없기 때문이다.

baseline은 이렇게 묻는다.

- 가장 단순한 모델보다 얼마나 좋아졌는가?
- 개선 폭이 실제로 의미 있는가?
- 계산 비용이 늘어난 만큼 이득이 있는가?

### 5-2. 약한 baseline과 강한 baseline

이 트랙에서는 보통 두 단계를 둔다.

- 약한 baseline: Dummy / Linear / Logistic
- 강한 baseline: RandomForest / HistGBDT / XGBoost / tuned model

이 구조를 쓰는 이유는,
"문제를 풀 수 있는 최소 수준"과 "tabular strong baseline"을 함께 보기 위해서다.

---

## 6. metric 이론

metric은 단순한 점수가 아니라, **모델이 무엇을 잘하고 무엇을 못하는지 묻는 언어**다.

### 6-1. classification metric

#### Accuracy
가장 직관적이지만 imbalance에서 쉽게 속는다.

예를 들어 positive가 10%인 데이터에서 모두 negative로 예측해도 accuracy는 90%가 될 수 있다.

#### Precision / Recall / F1

- Precision: positive라고 한 것 중 실제 positive 비율
- Recall: 실제 positive 중 찾아낸 비율
- F1: precision과 recall의 균형

이 metric이 나온 이유는,
"맞춘 개수"만으로는 positive 탐지 능력을 알 수 없기 때문이다.

#### AUROC
threshold를 고정하지 않고 ranking 품질을 본다.
"positive를 negative보다 위에 두는 능력"을 묻는다.

#### AUPRC
positive가 희소할 때 더 중요하다.
실제로 rare positive를 잘 끌어올리는지 보기에 적합하다.

### 6-2. regression metric

#### MAE
오차의 절대값 평균이다.
해석이 쉽고 직관적이다.

#### RMSE
큰 오차를 더 강하게 벌점 준다.
즉, tail error에 더 민감하다.

#### R²
대략적으로 설명력의 비율을 보여 주지만,
단독으로 쓰면 오해하기 쉽다. 반드시 MAE/RMSE와 함께 본다.

### 6-3. multiclass metric

#### Macro-F1 / Macro-Recall
class마다 먼저 점수를 계산한 뒤 평균낸다.
그래서 큰 class가 작은 class를 덮어버리는 현상을 줄여 준다.

이 metric이 중요한 이유는,
대규모 multiclass 문제에서 accuracy가 높아도 소수 class는 망가질 수 있기 때문이다.

---

## 7. 결과 figure와 분석 figure는 왜 구분하는가

리포트에는 보통 두 종류의 figure가 있다.

### 7-1. Results figure

질문: **그래서 성능이 어땠는가?**

예:
- ROC/PR curve
- confusion matrix
- parity plot
- residual plot
- metric-vs-time

### 7-2. Analysis figure

질문: **왜 그런 결과가 나왔는가?**

예:
- permutation importance
- slice metric
- calibration curve
- failure case panel
- throughput bottleneck summary

좋은 실험은 result figure만 많고 analysis가 빈약한 실험이 아니다.
오히려 analysis figure가 있어야 다음 실험 가설이 생긴다.

---

## 8. failure analysis 이론

많은 초보 실험은 "점수 상승"까지만 기록한다.
하지만 실제 학습은 **틀린 샘플을 읽는 순간** 시작된다.

failure analysis에서 봐야 할 것:

- 특정 demographic / geography / weather slice에 오차가 몰리는가?
- high-confidence error가 존재하는가?
- target high-end / low-end에서 systematic bias가 있는가?
- class confusion이 반복되는 pair가 있는가?

즉, failure analysis는 단순한 부록이 아니라 다음 실험 설계의 출발점이다.

---

## 9. 모델 해석 이론

### 9-1. feature importance는 왜 필요한가

점수가 좋아도 모델이 이상한 feature에 과하게 의존하면 위험하다.

예를 들어:
- leakage feature를 학습했을 수 있다.
- spurious correlation을 잡았을 수 있다.
- 특정 slice에만 유리한 feature를 과하게 사용했을 수 있다.

### 9-2. permutation importance

feature를 섞어 성능이 얼마나 떨어지는지 본다.
이는 "이 feature가 실제 예측에 얼마나 기여했는가"를 모델-불문으로 비교하기 좋다.

하지만 주의점도 있다.
- correlation이 높은 feature끼리는 importance가 분산될 수 있다.
- 중요한 feature가 낮게 보일 수도 있다.

즉, importance는 진실이 아니라 **해석 도구**다.

---

## 10. 비용-성능 trade-off 이론

특히 large-scale stage에서는 성능만 높으면 끝이 아니다.

같이 봐야 하는 것:
- fit time
- predict time
- peak memory
- throughput
- GPU 사용 여부

왜냐하면 실제 운영에서는 "조금 더 높은 점수"보다 "훨씬 더 빠르고 싸게 학습/추론되는 모델"이 더 가치 있을 수 있기 때문이다.

---

## 11. 이 트랙을 공부할 때 계속 던져야 할 질문

1. 이 split은 문제 정의와 맞는가?
2. 이 preprocessing은 leakage 없이 적용됐는가?
3. baseline 대비 개선폭이 충분한가?
4. primary metric은 왜 이것인가?
5. result figure와 analysis figure가 같은 이야기를 하는가?
6. 대표 failure case를 실제로 읽어봤는가?
7. 다음 실험 가설은 metric/figure 관찰에서 자연스럽게 나오는가?

---

## 12. 다음 단계와의 연결

이 ML 트랙의 이론은 이후 단계에서 그대로 확장된다.

- NLP에서도 split/leakage/metric 해석은 그대로 중요하다.
- Multimodal에서도 calibration, slice analysis, failure case는 그대로 중요하다.
- 모델만 바뀌고, **좋은 실험의 원칙은 그대로 유지된다.**

즉, `01_ml`은 쉬운 단계가 아니라, 이후 모든 실험의 공통 문법을 배우는 단계다.
