# 01. 표형 분류 한눈 요약

## 1. 한 줄 결론

Adult Census Income 분류에서는 `random_forest` 가 **AUPRC 기준으로 가장 안정적**이었다.
`gpu_mlp` 는 accuracy 가 더 높았지만 ranking 품질은 그보다 약했고, 이 차이는 “정답 비율” 과 “positive 탐지 능력” 이 다르다는 점을 보여 준다.

- 과제: Adult Census Income 이진 분류
- 최고 모델: `random_forest`
- 핵심 지표: `auprc`=0.7834, `auroc`=0.9105, `f1`=0.6971, `accuracy`=0.8354
- 해석: positive class 가 희소하므로 accuracy 보다 AUPRC 와 calibration 을 더 중요하게 봐야 한다

---

## 2. 왜 이론 문서가 필요한가

- 상세 이론 문서: [THEORY.md](../../../../01_ml/01_tabular_classification/THEORY.md)
- 자세한 설명이 있는 README: [01_ml/01_tabular_classification/README.md](../../../../01_ml/01_tabular_classification/README.md)

이 실험에서 기억해야 할 이론은 다음과 같다.

- accuracy 는 majority class 에 쉽게 속는다.
- AUPRC 는 희소한 positive class 의 ranking 품질을 잘 보여 준다.
- threshold 를 바꾸면 F1 과 recall 이 달라진다.
- calibration 은 confidence 를 신뢰할 수 있는지 알려 준다.
- slice 분석은 model bias 를 드러낸다.

---

## 3. 모델 비교

| 모델 | AUPRC | AUROC | F1 | ACCURACY | FIT_SEC |
| --- | --- | --- | --- | --- | --- |
| `random_forest` | 0.7834 | 0.9105 | 0.6971 | 0.8354 | 1.70 |
| `logistic_regression` | 0.7657 | 0.9044 | 0.6724 | 0.8055 | 4.36 |
| `gpu_mlp` | 0.7569 | 0.9021 | 0.6851 | 0.8510 | 4.48 |
| `dummy_prior` | 0.2407 | 0.5000 | 0.0000 | 0.7593 | 0.00 |

### 읽는 법

- `dummy_prior` 는 majority class baseline 이다. accuracy 는 높아 보이지만 AUPRC 가 매우 낮다.
- `logistic_regression` 은 강한 선형 baseline 이다.
- `random_forest` 는 비선형 상호작용을 잘 잡아서 ranking 품질이 가장 좋았다.
- `gpu_mlp` 는 accuracy 는 높았지만 AUPRC 는 random_forest 보다 낮았다.

---

## 4. 결과 해석 / 실패 분석

- 클래스 분포는 `<=50K` 24,720개, `>50K` 7,841개로 불균형하다.
- 고확신 오답 30개 중 `Male` 이 29개, `Female` 이 1개였다.
- 상위 오답의 학력은 주로 `Bachelors`, `Masters`, `Doctorate` 였다.
- 평균 연령은 44.9세, 평균 근무시간은 47.9시간이었다.
- `error_slice_by_sex` 에서는 Male error rate 가 0.210, Female 이 0.072 로 보였다.

이 패턴은 모델이 **고학력 + 긴 근무시간 + 특정 직업** 을 고소득 신호로 강하게 학습했지만, 그 archetype 이 항상 정답은 아니라는 뜻이다.
즉, 모델은 평균적으로는 잘 맞지만, 특정 slice 와 high-confidence 예외에서 과신한다.

---

## 5. 결과 Figure

### PR curve

- [pr_curve.svg](figures/results/pr_curve.svg)

희소 positive 를 얼마나 잘 찾는지 본다.

### ROC curve

- [roc_curve.svg](figures/results/roc_curve.svg)

threshold 전반에서 ranking 이 얼마나 좋은지 본다.

### Confusion matrix

- [confusion_matrix.svg](figures/results/confusion_matrix.svg)

FP 와 FN 의 균형을 본다.

### Calibration curve

- [calibration_curve.svg](figures/results/calibration_curve.svg)

확률을 얼마나 믿을 수 있는지 본다.

### Class distribution

- [class_distribution.svg](figures/results/class_distribution.svg)

baseline 이 왜 높아 보이는지 이해하게 해 준다.

---

## 6. 분석 Figure

### Permutation importance

- [permutation_importance.svg](figures/analysis/permutation_importance.svg)

어떤 feature 가 예측에 많이 쓰였는지 본다.

### Error slice by sex

- [error_slice_by_sex.svg](figures/analysis/error_slice_by_sex.svg)

민감 속성 slice 에서 오차율 차이가 있는지 본다.

### Confidence vs correctness

- [confidence_vs_correctness.svg](figures/analysis/confidence_vs_correctness.svg)

confidence 가 높을수록 실제로도 맞는지 본다.

### Failure examples

- [failure_examples.svg](figures/analysis/failure_examples.svg)

고확신 오답이 어떤 feature 조합에 몰리는지 본다.

---

## 7. 다음 실험 가설

1. threshold 를 validation set 에서 튜닝하면 F1 을 더 끌어올릴 수 있다.
2. calibration 을 개선하면 고확신 오답을 줄일 수 있다.
3. class-weight 를 주면 positive 탐지 능력이 더 좋아질 수 있다.
4. boosting 계열 모델과 비교하면 ranking 품질이 더 좋아질 수 있다.
5. age / occupation slice 를 추가하면 다른 실패 패턴이 더 드러날 수 있다.

---

## 8. 리포트 링크

- 최신 상세 리포트: [README.md](README.md)
- 최신 요약 리포트: [summary.md](summary.md)
- 이론 문서: [THEORY.md](../../../../01_ml/01_tabular_classification/THEORY.md)
