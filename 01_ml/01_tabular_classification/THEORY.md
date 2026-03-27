# 01 Tabular Classification 이론 노트

이 문서는 “표형 분류를 왜 이렇게까지 자세히 공부해야 하는가?”라는 질문에 답하기 위해 존재한다.
단순히 모델을 하나 돌리고 점수만 확인하면 끝나는 문제가 아니라,

- 데이터가 어떤 구조를 가지는지,
- 왜 accuracy 하나로는 부족한지,
- threshold를 바꾸면 무엇이 달라지는지,
- 어떤 샘플에서 모델이 과신하는지,
- 그리고 그 실패가 특정 slice에 몰리는지

까지 함께 읽어야 표형 분류를 이해했다고 말할 수 있다.

이 노트는 `Adult Census Income` 실험을 공부할 때 보는 이론 메모다.
실험 결과는 최신 리포트에서 확인할 수 있다.

- [최신 리포트 README](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/README.md)
- [최신 리포트 요약](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/summary.md)

---

## 1. 이 이론 문서가 왜 생겼는가

표형 분류는 처음 보면 매우 단순해 보인다.
숫자형과 범주형 feature를 넣고, 분류기를 학습시키고, accuracy를 보면 되는 것처럼 느껴진다.

하지만 실제로는 아래 문제가 한 번에 터진다.

1. **클래스 불균형** 때문에 accuracy가 쉽게 속인다.
2. **threshold** 때문에 같은 score도 다른 label이 된다.
3. **ranking 품질**과 **label 정확도**가 서로 다르다.
4. **calibration** 이 맞지 않으면 확률을 믿기 어렵다.
5. **slice별 오차**를 보면 전체 평균이 가리는 실패가 드러난다.

그래서 이 문서는 “모델 하나를 돌리는 법”이 아니라,
**분류 실험을 해석하는 법**을 정리하려고 만들어졌다.

---

## 2. 이 문제가 실제로 무엇을 푸는가

이번 실습의 대상은 `scikit-learn/adult-census-income` 이진 분류 문제다.

- 입력: 나이, 학력, 직업, 근로시간, 자본 이득/손실, 결혼 상태 등
- 출력: `<=50K` / `>50K`

겉보기에는 단순하지만 실제로는 복합적이다.

- 수치형과 범주형이 섞여 있다.
- 결측치가 `?` 형태로 들어 있다.
- `>50K` 는 소수 클래스다.
- 성별, 직업, 학력, 근로시간 조합에 따라 오차 패턴이 달라진다.

즉, 이 문제는 “분류가 되는가?”보다
**“어떤 정보가 성능에 기여하고, 어디에서 틀리는가?”**를 배우는 데 더 적합하다.

---

## 3. 분류에서 가장 먼저 구분해야 할 것

### 3.1 score와 label은 다르다

모델은 먼저 score 또는 확률을 내고, 그다음 threshold로 label을 만든다.

- score: “양성일 가능성”에 대한 연속값
- label: threshold를 통과한 뒤의 최종 판단

이 둘은 같은 것이 아니다.
`0.83` 같은 값은 “양성일 가능성이 높다”는 뜻이지, 아직 최종 정답은 아니다.

### 3.2 ranking과 threshold는 다르다

같은 score라도 threshold를 바꾸면 FP/FN이 달라진다.

- threshold를 높이면 precision은 오르기 쉽고 recall은 떨어지기 쉽다.
- threshold를 낮추면 recall은 오르기 쉽고 precision은 떨어지기 쉽다.

따라서

- **ranking** 이 좋은 모델
- **특정 threshold에서 label이 잘 맞는 모델**

은 서로 다를 수 있다.

### 3.3 calibration은 confidence가 믿을 만한지 본다

calibration은 “예측 확률 0.9를 준 샘플이 정말 90% 정도 맞는가?”를 묻는다.

ranking이 좋아도 calibration이 나쁘면 운영 환경에서 위험하다.
왜냐하면 모델이 **틀리면서도 매우 자신 있게 말할 수 있기** 때문이다.

---

## 4. 이 실습에서 왜 baseline이 중요했는가

Adult 데이터는 양성 클래스가 상대적으로 희소하다.
실제 분포를 보면 클래스 분포는 대략 다음과 같다.

- `<=50K`: 24,720
- `>50K`: 7,841

즉, 전체의 약 75.9%가 `<=50K` 이다.
그래서 majority class만 찍어도 accuracy가 높게 보인다.

이때 필요한 것이 baseline이다.

### 4.1 DummyClassifier

- “아무것도 안 했을 때”의 성능을 보여 준다.
- accuracy는 높아 보일 수 있지만 positive를 거의 못 찾는다.
- 이번 실험에서 `dummy_prior` 는 accuracy 0.7593, AUPRC 0.2407 이었다.

이 수치는 “accuracy만 보면 안 된다”는 사실을 가장 직관적으로 보여 준다.

### 4.2 Logistic Regression

- 가장 기본적인 선형 baseline이다.
- 해석이 쉽고 빠르다.
- tabular에서는 생각보다 강하다.

이번 실험에서는 AUPRC 0.7657, AUROC 0.9044, F1 0.6724 로 꽤 강한 기준선이었다.

### 4.3 Random Forest

- 비선형 상호작용을 잘 잡는다.
- 범주형 one-hot 이후 tabular에서 자주 강하다.
- 이번 실험의 best model 이다.

### 4.4 GPU MLP

- 비선형 표현력이 높다.
- GPU 사용 연습에는 좋다.
- 하지만 tabular에서는 tree ensemble이 더 안정적인 경우가 많다.

이번 실험에서는 accuracy가 가장 높았지만, AUPRC는 random_forest 보다 낮았다.
즉, **정답 비율과 positive 탐지 능력은 다르다**는 점이 다시 확인됐다.

---

## 5. 전처리는 왜 이렇게 구성하는가

### 5.1 누수 방지

가장 흔한 실수는 train 밖의 정보를 전처리에 섞는 것이다.

그래서

- imputer,
- encoder,
- scaler

는 pipeline 안에서 train에만 fit 해야 한다.

### 5.2 범주형과 수치형은 분리해서 다뤄야 한다

Adult 데이터는 다음처럼 feature 타입이 섞여 있다.

- 수치형: `age`, `hours.per.week`, `capital.gain`, `capital.loss`
- 범주형: `workclass`, `education`, `occupation`, `marital.status`

따라서 `ColumnTransformer` 같은 구조가 필요하다.

- 수치형: 결측 대체, 필요 시 스케일링
- 범주형: one-hot encoding

### 5.3 `?` 는 그냥 문자로 두면 안 된다

이 데이터셋에서 `?` 는 실제 범주라기보다 missing sentinel 에 가깝다.
그러므로 먼저 결측으로 정규화한 뒤 적절히 처리하는 것이 자연스럽다.

---

## 6. 이 실험의 메트릭을 어떻게 읽어야 하는가

이번 리포트의 대표 수치는 다음과 같다.

| 모델 | AUPRC | AUROC | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| `random_forest` | 0.7834 | 0.9105 | 0.6971 | 0.8354 |
| `logistic_regression` | 0.7657 | 0.9044 | 0.6724 | 0.8055 |
| `gpu_mlp` | 0.7569 | 0.9021 | 0.6851 | 0.8510 |
| `dummy_prior` | 0.2407 | 0.5000 | 0.0000 | 0.7593 |

이 표는 다음처럼 읽는다.

1. `dummy_prior` 는 majority class baseline 이다.
   accuracy가 높아도 positive ranking 은 거의 못 한다.
2. `logistic_regression` 은 선형 baseline 이지만 이미 꽤 강하다.
3. `random_forest` 는 ranking 품질과 threshold 성능을 모두 잘 잡았다.
4. `gpu_mlp` 는 accuracy가 가장 높았지만 AUPRC 는 그보다 낮았다.

핵심은 **하나의 숫자가 전체 성능을 설명하지 못한다**는 점이다.

---

## 7. 결과 그림을 어떻게 읽어야 하는가

### 7.1 class distribution

- [class_distribution.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/class_distribution.svg)

이 그림은 왜 accuracy가 쉽게 높아 보이는지 설명한다.
`<=50K` 가 훨씬 많기 때문에 baseline이 강해 보인다.

### 7.2 PR curve

- [pr_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/pr_curve.svg)

희소한 positive class 를 얼마나 잘 찾는지 보여 준다.
이 실습에서는 AUPRC 해석이 매우 중요하다.

### 7.3 ROC curve

- [roc_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/roc_curve.svg)

전체 ranking quality 를 본다.
다만 클래스 불균형이 큰 문제에서는 PR curve 와 같이 봐야 한다.

### 7.4 confusion matrix

- [confusion_matrix.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/confusion_matrix.svg)

best model 인 random_forest 의 혼동행렬은 다음과 같이 읽힌다.

- TN: 3,156
- FP: 553
- FN: 251
- TP: 925

이 값들은 precision/recall/F1 이 어디서 나왔는지 직접 보여 준다.

### 7.5 calibration curve

- [calibration_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/calibration_curve.svg)

예측 확률이 높아질수록 실제 정답률도 함께 올라가는지 확인한다.
이번 실험에서는 전반적으로 상승 추세가 보이지만, 완벽하게 ideal line 을 따르지는 않는다.
즉, confidence 는 참고할 수 있지만 무조건 믿으면 안 된다.

---

## 8. 실패 패턴은 무엇을 말해 주는가

실패 분석은 이론의 핵심이다.
정확도만 보면 안 되는 이유가 여기서 드러난다.

### 8.1 class imbalance 에서 생기는 기본 오해

전체 분포가 `<=50K` 쪽으로 크게 치우쳐 있기 때문에,
모델은 자연스럽게 majority class 를 잘 맞추는 방향으로도 높은 accuracy 를 얻을 수 있다.

하지만 positive class 를 놓치면 실무적 가치가 낮다.
그래서 AUPRC, recall, calibration 을 함께 봐야 한다.

### 8.2 error slice by sex

- [error_slice_by_sex.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/error_slice_by_sex.svg)

이 그림에서는 mean error rate 가

- Male: 0.210
- Female: 0.072

로 보인다.

즉, 평균적으로는 **Male slice 에서 오차가 더 많이 발생**한다.
이 사실은 전체 점수만으로는 보이지 않는다.

### 8.3 failure examples

- [failure_examples.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/failure_examples.svg)

이 그림은 high-confidence failure 를 직접 보여 준다.
대표적인 오답은 다음과 같은 공통점을 가진다.

- 나이: 대체로 37~55세
- 학력: `Doctorate`, `Masters`, `Bachelors`
- 직업: `Exec-managerial`, `Prof-specialty`
- 근로시간: 45~70시간
- 실제 라벨: `<=50K`
- 예측: `>50K`
- score: 0.927~0.957 수준의 고확신

예를 들어 다음과 같은 패턴이 반복된다.

- `45 / Doctorate / Exec-managerial / 60h / true <=50K / pred >50K / score 0.957`
- `55 / Masters / Exec-managerial / 60h / true <=50K / pred >50K / score 0.947`
- `46 / Doctorate / Prof-specialty / 70h / true <=50K / pred >50K / score 0.946`

이건 모델이

> “고학력 + 긴 근무시간 + 특정 직업 = 고소득”

이라는 archetype을 강하게 학습했음을 뜻한다.
문제는 이 archetype이 항상 정답은 아니라는 점이다.
그래서 **예외 케이스에서 과신한 오답**이 나온다.

### 8.4 confidence vs correctness

- [confidence_vs_correctness.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/confidence_vs_correctness.svg)

이 그림은 confidence bin 이 올라갈수록 실제 accuracy 도 올라가는지를 본다.
이번 실험에서는 전반적 상승 추세는 있지만, 일부 구간에서 흔들림이 있어 완벽한 calibration 은 아니다.

즉, 모델이 높은 확률을 준다고 해서 항상 맞는 것은 아니다.

---

## 9. 왜 random_forest 가 가장 좋았는가

이 데이터는 feature 들이 서로 독립적이지 않다.

- 학력
- 직업
- 결혼 상태
- 근로시간
- 자본 이득/손실

이 변수들은 함께 작동한다.
random_forest 는 이런 비선형 상호작용을 잘 포착하므로 AUPRC 에서 가장 좋은 결과를 낸 것으로 해석할 수 있다.

반대로 `gpu_mlp` 는 accuracy 가 높았지만 positive ranking 이 더 안정적이지 않았다.
즉, **threshold 하나에서는 잘 맞아도, 전체 ranking 은 더 나쁠 수 있다**.

---

## 10. 다음 실험 가설

이론을 공부했다면 다음 질문으로 넘어가야 한다.

1. **threshold tuning**
   validation set 에서 threshold 를 조정하면 F1 과 recall 의 균형이 좋아질 수 있다.

2. **calibration 개선**
   Platt scaling 또는 isotonic calibration 으로 고확신 오답을 줄일 수 있다.

3. **class weight / cost-sensitive learning**
   positive class 를 더 중요하게 보면 AUPRC 와 recall 이 개선될 수 있다.

4. **boosting 계열 비교**
   random_forest 다음에는 gradient boosting 계열이 더 좋은 ranking 을 줄 가능성이 있다.

5. **slice 확대**
   sex 외에도 age, occupation, education, hours.per.week slice 를 보면 더 많은 실패 패턴이 드러난다.

---

## 11. 한 문장 요약

**표형 분류에서는 accuracy 하나보다, positive를 얼마나 잘 찾는지, 확률을 얼마나 믿을 수 있는지, 그리고 어떤 slice 에서 실패하는지를 같이 읽어야 한다.**
