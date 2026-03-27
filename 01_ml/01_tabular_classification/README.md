# 01 Tabular Classification

## 0. 이 stage가 왜 존재하는가

이 stage의 목표는 단순히 Adult 데이터를 분류하는 것이 아니다.
**표형 분류를 해석하는 방식 자체**를 몸에 익히는 것이다.

표형 분류에서는 모델을 돌리는 것보다 다음 질문을 읽는 일이 더 중요하다.

- 왜 accuracy가 쉽게 높아 보이는가?
- 왜 threshold 를 바꾸면 결과가 달라지는가?
- 왜 AUPRC 와 AUROC 를 함께 봐야 하는가?
- 왜 전처리를 pipeline 안에서만 fit 해야 하는가?
- 왜 실패 사례를 봐야 진짜 공부가 되는가?

이 stage는 그 질문에 답하는 연습이다.

---

## 1. 먼저 읽어야 할 문서

- 이론 문서: [THEORY.md](THEORY.md)
- 최신 실험 리포트: [reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/README.md](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/README.md)
- 한눈 요약: [reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/summary.md](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/summary.md)

이 stage의 이해 순서는 보통 다음이 가장 좋다.

1. `THEORY.md` 로 개념을 잡는다.
2. 최신 리포트 README 로 숫자와 figure 를 읽는다.
3. summary.md 로 핵심만 다시 정리한다.

---

## 2. 이 stage에서 다루는 문제

이번 과제의 기준 데이터셋은 `scikit-learn/adult-census-income` 이다.

```python
from datasets import load_dataset

ds = load_dataset("scikit-learn/adult-census-income", split="train")
df = ds.to_pandas()
```

### 왜 이 데이터인가

이 데이터는 표형 분류에서 자주 만나는 함정을 한 번에 담고 있다.

- 범주형과 수치형 feature 가 함께 있다.
- 결측이 `?` 형태로 존재한다.
- 클래스 불균형이 있다.
- 민감 속성 slice 를 함께 볼 수 있다.
- 단순 accuracy 로는 충분하지 않다.

즉, “모델을 하나 돌려보는 데이터”가 아니라
**분류 실험 문법을 공부하기 좋은 데이터**다.

### debug fallback

- `Breast Cancer Wisconsin (Diagnostic)`

---

## 3. 이 stage에서 꼭 익혀야 하는 개념

### 3.1 score 와 label 은 다르다

모델은 먼저 score 를 내고, 그 다음 threshold 로 label 을 만든다.
따라서 점수 품질과 label 품질을 분리해서 봐야 한다.

### 3.2 불균형 데이터에서는 accuracy 가 쉽게 속인다

majority class 비중이 높으면 dummy baseline 도 accuracy 가 높아 보인다.
그래서 AUPRC, F1, calibration 을 같이 봐야 한다.

### 3.3 전처리는 train 에서만 fit 해야 한다

imputer, encoder, scaler 는 pipeline 안에 넣어야 한다.
train 밖의 정보를 fit 단계에 섞으면 data leakage 가 발생한다.

### 3.4 모델마다 강점이 다르다

- Logistic Regression: 해석이 쉽고 강한 선형 baseline
- Random Forest: 비선형 상호작용을 잘 잡는 strong baseline
- GPU MLP: 표현력은 높지만 tabular 에서 항상 이기지는 않음

### 3.5 실패 사례를 봐야 한다

전체 평균이 좋아도 특정 slice 에서 과신한 오답이 반복될 수 있다.
이럴 때는 confusion matrix, slice plot, failure examples 를 같이 본다.

---

## 4. 실습 파이프라인

이번 stage 에서는 보통 다음 순서로 실험을 읽는다.

1. 데이터 카드 확인
2. train / valid / test split 과 seed 고정
3. 수치형 / 범주형 컬럼 분리
4. `ColumnTransformer` 기반 전처리 파이프라인 구축
5. `DummyClassifier` 와 `LogisticRegression` 으로 baseline 확인
6. `RandomForest` 또는 `HistGradientBoosting` 으로 strong baseline 확인
7. AUROC, AUPRC, F1, calibration 비교
8. failure examples 와 slice error 분석

이 순서의 핵심은 “모델 성능표를 보는 것”이 아니라
**성능표가 왜 그렇게 나왔는지 설명할 수 있게 되는 것**이다.

---

## 5. 결과를 읽는 기준

최신 결과는 아래 리포트에서 확인한다.

- [최신 리포트 README](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/README.md)
- [최신 리포트 summary](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/summary.md)

리포트를 읽을 때는 다음 질문을 같이 보자.

1. 왜 `random_forest` 가 가장 좋은 AUPRC 를 냈는가?
2. 왜 `gpu_mlp` 는 accuracy 가 더 높았는데 ranking 품질은 덜 안정적인가?
3. 왜 고확신 오답이 특정 slice 에 몰리는가?
4. calibration curve 가 confidence 를 얼마나 믿을 수 있게 보여 주는가?

---

## 6. 최신 리포트에서 같이 봐야 할 figure

### 결과 figure

- [class_distribution.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/class_distribution.svg)
- [pr_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/pr_curve.svg)
- [roc_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/roc_curve.svg)
- [confusion_matrix.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/confusion_matrix.svg)
- [calibration_curve.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/results/calibration_curve.svg)

### 분석 figure

- [permutation_importance.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/permutation_importance.svg)
- [error_slice_by_sex.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/error_slice_by_sex.svg)
- [confidence_vs_correctness.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/confidence_vs_correctness.svg)
- [failure_examples.svg](../../reports/01_ml/01_tabular_classification/20260326-172429_adult-census-income_model-suite_s42/figures/analysis/failure_examples.svg)

---

## 7. 이 stage의 핵심 해석 포인트

### 7.1 dummy baseline 이 높아 보여도 놀라지 말 것

클래스 분포가 한쪽으로 치우치면 dummy baseline accuracy 가 높아 보인다.
그렇다고 해서 모델이 positive 를 잘 찾는다는 뜻은 아니다.

### 7.2 AUPRC 는 희소 positive 에서 특히 중요하다

Adult income 처럼 양성 비율이 낮은 문제에서는 AUPRC 가 ranking 품질을 더 잘 보여 준다.
그래서 accuracy 만 보면 실험을 잘못 해석할 수 있다.

### 7.3 threshold 와 ranking 을 분리해서 볼 것

같은 모델이라도 threshold 를 바꾸면 F1 과 recall 이 달라진다.
따라서 score 전체와 특정 threshold 의 label 성능을 따로 읽어야 한다.

### 7.4 calibration 과 failure analysis 는 필수다

높은 score 를 줬다고 항상 맞는 것은 아니다.
고확신 오답이 어디에 몰리는지, confidence 가 실제 correctness 와 같이 움직이는지 꼭 확인해야 한다.

---

## 8. 승격 기준

이 stage 는 다음 질문에 답할 수 있어야 다음 단계로 넘어갈 수 있다.

1. baseline 대비 개선이 분명한가?
2. 데이터 누수 가능성이 제거되었는가?
3. 결과 figure 만 봐도 왜 성능이 달라졌는지 설명할 수 있는가?
4. 고확신 오답을 실제로 확인했는가?
5. 다음 실험 가설을 문장으로 쓸 수 있는가?

---

## 9. 이 stage에서 특히 기억할 문장

- accuracy 하나만으로는 충분하지 않다.
- AUPRC 는 희소 positive class 에서 매우 중요하다.
- calibration 은 고확신 예측을 믿을 수 있는지 보여 준다.
- 실패 사례를 봐야 진짜 공부가 된다.
