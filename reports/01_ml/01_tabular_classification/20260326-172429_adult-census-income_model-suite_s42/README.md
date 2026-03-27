# 01. 표형 분류 결과 리포트

## 1. 한 줄 결론

Adult Census Income 분류에서는 **`random_forest` 가 AUPRC 기준으로 가장 안정적**이었다.
다만 `gpu_mlp` 는 accuracy 가 더 높았고, 이 차이는 **“label 을 맞히는 능력”** 과 **“positive 를 잘 찾는 능력”** 이 다르다는 사실을 다시 보여 준다.

- 과제: Adult Census Income 이진 분류
- 최고 모델: `random_forest`
- 핵심 지표: `auprc`=0.7834, `auroc`=0.9105, `f1`=0.6971, `accuracy`=0.8354
- 해석: positive class 가 희소한 상황에서는 accuracy 보다 AUPRC 와 calibration 을 더 먼저 봐야 한다

---

## 2. 이론 문서와 같이 읽어야 하는 이유

- 상세 이론 문서: [THEORY.md](../../../../01_ml/01_tabular_classification/THEORY.md)
- stage README: [01_ml/01_tabular_classification/README.md](../../../../01_ml/01_tabular_classification/README.md)

이 리포트는 숫자만 모아 둔 문서가 아니다.
이론 문서에서 말한 개념이 실제 결과에서 어떻게 드러났는지 확인하는 공부 노트다.

특히 다음 세 가지를 같이 읽어야 한다.

1. **accuracy 는 majority class 에 쉽게 속는다.**
2. **AUPRC 는 희소 positive ranking 을 더 민감하게 보여 준다.**
3. **failure examples 와 slice plot 을 봐야 모델의 습관이 보인다.**

---

## 3. 실험 설정

- 데이터셋: `scikit-learn/adult-census-income`
- 분할: train / valid / test 고정 분할
- 전처리: 수치형 median imputation, 범주형 most-frequent imputation, one-hot encoding
- 비교 모델: `dummy_prior`, `logistic_regression`, `random_forest`, `gpu_mlp`
- 평가 지표: accuracy, precision, recall, F1, AUROC, AUPRC

이 실험의 목적은 “모델 하나를 돌려서 최고 점수를 보는 것”이 아니다.
목표는 **누수 없는 전처리 + baseline 비교 + 실패 분석**을 하나의 흐름으로 읽는 것이다.

---

## 4. 클래스 분포가 왜 중요한가

- [class_distribution.svg](figures/results/class_distribution.svg)

이 그림은 이 실험의 출발점이다.

- `<=50K`: 24,720
- `>50K`: 7,841

즉, 데이터는 대략 75.9% 대 24.1% 로 불균형하다.
그래서 `dummy_prior` 도 accuracy 0.7593 을 얻을 수 있었다.

하지만 이 수치는 “모델이 잘했다”는 뜻이 아니다.
오히려 **majority class 만 찍어도 accuracy 가 높아지는 문제 구조**를 보여 준다.

---

## 5. 모델 비교와 메트릭 해석

| 모델 | AUPRC | AUROC | F1 | ACCURACY | FIT_SEC |
| --- | --- | --- | --- | --- | --- |
| `random_forest` | 0.7834 | 0.9105 | 0.6971 | 0.8354 | 1.70 |
| `logistic_regression` | 0.7657 | 0.9044 | 0.6724 | 0.8055 | 4.36 |
| `gpu_mlp` | 0.7569 | 0.9021 | 0.6851 | 0.8510 | 4.48 |
| `dummy_prior` | 0.2407 | 0.5000 | 0.0000 | 0.7593 | 0.00 |

### 메트릭을 어떻게 읽어야 하는가

- `dummy_prior`
  - majority class baseline 이다.
  - accuracy 는 높아 보여도 AUPRC 0.2407 은 positive 를 거의 못 찾는다는 뜻이다.

- `logistic_regression`
  - 선형 baseline 으로 이미 꽤 강하다.
  - 이 데이터가 완전히 선형은 아니지만, 전처리를 잘하면 의미 있는 성능이 나온다.

- `random_forest`
  - AUPRC 와 AUROC 가 가장 좋았다.
  - 이 데이터에는 단순 선형 경계보다 feature 간 비선형 상호작용이 중요하다고 읽을 수 있다.

- `gpu_mlp`
  - accuracy 는 가장 높았지만 AUPRC 는 random_forest 보다 낮았다.
  - 따라서 “정답 비율” 은 좋지만 “positive 를 높은 순서로 정렬하는 능력” 은 더 안정적이지 않았다고 보는 편이 맞다.

즉, 성능표는 한 줄로 끝나지 않는다.
**어떤 메트릭이 무엇을 측정하는지**를 분리해서 읽어야 한다.

---

## 6. 결과 해석 / 실패 분석

### 6.1 왜 random_forest 가 가장 강했는가

Adult 데이터는 다음 feature 들이 서로 얽혀 있다.

- `education`
- `education.num`
- `hours.per.week`
- `capital.gain`
- `capital.loss`
- `occupation`
- `marital.status`

이 조합은 독립적이지 않다.
random_forest 는 이런 비선형 조합을 잘 잡기 때문에 ranking 품질에서 가장 좋은 결과가 나왔다고 해석할 수 있다.

### 6.2 왜 `gpu_mlp` 는 accuracy 가 높았는데 AUPRC 는 더 낮았는가

accuracy 는 threshold 기준 label 정확도다.
반면 AUPRC 는 score ranking 전체를 본다.

즉 `gpu_mlp` 는 threshold 하나에서는 더 잘 맞췄지만, positive 를 높은 순서로 정렬하는 능력은 random_forest 보다 덜 안정적이었다.

### 6.3 고확신 오답은 어디에 몰렸는가

- [failure_examples.svg](figures/analysis/failure_examples.svg)

이 그림은 best model 이 어떤 샘플을 헷갈렸는지 직접 보여 준다.
대표적인 고확신 오답은 다음 패턴을 가진다.

- 나이: 대체로 37~55세
- 학력: `Doctorate`, `Masters`, `Bachelors`
- 직업: `Exec-managerial`, `Prof-specialty`
- 근로시간: 45~70시간
- 실제 라벨: `<=50K`
- 예측: `>50K`
- score: 0.927~0.957 수준의 고확신

예시 몇 개만 봐도 패턴이 또렷하다.

- `45 / Doctorate / Exec-managerial / 60h / true <=50K / pred >50K / score 0.957`
- `55 / Masters / Exec-managerial / 60h / true <=50K / pred >50K / score 0.947`
- `46 / Doctorate / Prof-specialty / 70h / true <=50K / pred >50K / score 0.946`

이 패턴은 모델이

> 고학력 + 긴 근무시간 + 특정 직업 = 고소득

이라는 archetype 을 강하게 학습했음을 뜻한다.
문제는 이 archetype 이 항상 정답은 아니라는 점이다.
그래서 **예외 케이스에서 과신한 오답**이 생긴다.

### 6.4 성별 slice 에서는 어떤 차이가 보였는가

- [error_slice_by_sex.svg](figures/analysis/error_slice_by_sex.svg)

이 그림에서는 mean error rate 가 다음처럼 보인다.

- Male: 0.210
- Female: 0.072

즉, **Male slice 에서 오차가 더 많이 발생**했다.
전체 평균만 보면 보이지 않는 차이이므로, slice 분석이 꼭 필요하다.

### 6.5 confidence 는 얼마나 믿을 수 있는가

- [confidence_vs_correctness.svg](figures/analysis/confidence_vs_correctness.svg)

confidence bin 이 올라갈수록 실제 accuracy 도 함께 올라가는지 확인한다.
이번 실험에서는 전반적 상승 추세가 있긴 하지만 완벽하지는 않다.

즉, 모델이 높은 확률을 준다고 해서 항상 맞는 것은 아니다.
고확신 오답이 존재하므로 calibration 을 계속 점검해야 한다.

### 6.6 confusion matrix 가 알려 주는 것

- [confusion_matrix.svg](figures/results/confusion_matrix.svg)

best model 인 random_forest 의 혼동행렬은 다음과 같다.

- TN: 3,156
- FP: 553
- FN: 251
- TP: 925

이 숫자는 precision/recall/F1 이 어디서 나왔는지 직접 보여 준다.
특히 FN 이 남아 있다는 것은 positive 를 완전히 놓치지 않았지만, 아직 더 잡아낼 여지가 있다는 뜻이다.

---

## 7. 결과 figure 를 읽는 법

### PR curve

- [pr_curve.svg](figures/results/pr_curve.svg)

positive class 가 희소할 때 모델이 얼마나 잘 찾는지 본다.
이 실험에서는 AUPRC 해석이 핵심이다.

### ROC curve

- [roc_curve.svg](figures/results/roc_curve.svg)

threshold 전체에 걸친 ranking quality 를 본다.
하지만 클래스 불균형이 큰 문제에서는 PR curve 와 같이 읽어야 한다.

### Calibration curve

- [calibration_curve.svg](figures/results/calibration_curve.svg)

예측 확률이 높을수록 실제 정답률도 올라가는지 확인한다.
이번 결과는 완벽한 ideal line 은 아니고, confidence 가 일부 구간에서 흔들린다.

### Class distribution

- [class_distribution.svg](figures/results/class_distribution.svg)

baseline 이 왜 높게 보이는지 설명하는 가장 첫 번째 그림이다.

---

## 8. 분석 figure 를 읽는 법

### Permutation importance

- [permutation_importance.svg](figures/analysis/permutation_importance.svg)

어떤 feature 를 섞었을 때 성능이 얼마나 떨어지는지 본다.
즉, 모델이 무엇을 중요하게 봤는지 해석할 수 있게 해 준다.

### Error slice by sex

- [error_slice_by_sex.svg](figures/analysis/error_slice_by_sex.svg)

성별 slice 에서 오차율 차이가 있는지 본다.
이번 실험에서는 Male 쪽 error rate 가 더 높았다.

### Confidence vs correctness

- [confidence_vs_correctness.svg](figures/analysis/confidence_vs_correctness.svg)

confidence 가 올라갈수록 correctness 도 같이 올라가는지 본다.
이 관계가 약하면 모델이 과신하고 있을 가능성이 있다.

### Failure examples

- [failure_examples.svg](figures/analysis/failure_examples.svg)

고확신 오답의 대표 표본을 모아 둔 그림이다.
정량 지표만으로는 놓치기 쉬운 오차 패턴을 직접 보여 준다.

---

## 9. 이 리포트가 남긴 해석

이 stage 의 핵심은 “random_forest 가 좋았다” 한 줄이 아니다.
핵심은 아래 세 문장이다.

1. **positive class 가 희소할 때는 accuracy 만 보면 안 된다.**
2. **threshold 기반 성능과 ranking 기반 성능은 다르다.**
3. **고확신 오답을 보면 모델이 어떤 편향을 학습했는지 보인다.**

즉, 성능표를 보는 것이 끝이 아니라, 성능표를 통해
**데이터 구조와 모델의 습관을 읽는 것**이 이 실습의 목적이다.

---

## 10. 다음 실험 가설

다음 단계는 단순히 모델을 바꾸는 것이 아니라,
**왜 틀리는지에 대한 가설**을 검증하는 것이다.

### 가설 1. threshold tuning

validation set 에서 threshold 를 튜닝하면 F1 과 recall 균형을 더 좋게 만들 수 있다.

### 가설 2. calibration 개선

Platt scaling 이나 isotonic calibration 으로 고확신 오답을 줄일 수 있다.

### 가설 3. class-weight / cost-sensitive learning

positive class 를 더 중요하게 두면 AUPRC 와 recall 이 개선될 수 있다.

### 가설 4. gradient boosting 비교

random_forest 다음에는 boosting 계열이 더 좋은 ranking 을 줄 수 있다.

### 가설 5. slice 확대

성별뿐 아니라 age, education, occupation, hours.per.week slice 를 함께 보면 더 많은 실패 패턴이 드러날 수 있다.

---

## 11. 연결 링크

- 이 stage 의 이론 문서: [THEORY.md](../../../../01_ml/01_tabular_classification/THEORY.md)
- 이 stage 의 README: [01_ml/01_tabular_classification/README.md](../../../../01_ml/01_tabular_classification/README.md)
- 이 stage 의 요약본: [summary.md](summary.md)
