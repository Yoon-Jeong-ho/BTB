# 01 Tabular Classification 이론 노트

이 문서는 Stage 1 에서 쓰는 용어와 개념을 **정리해 두는 사전**이자, 실험을 읽기 전에 먼저 이해해야 하는 **강의 노트**다.
핵심은 “분류를 했다”가 아니라 **분류 결과를 해석하는 틀을 갖추는 것**이다.

- stage 가이드: [README.md](README.md)
- 최신 artifact 리포트: [README.md](artifacts/20260327-164446_adult-census-income_model-suite_s42/README.md)
- 최신 artifact 요약: [summary.md](artifacts/20260327-164446_adult-census-income_model-suite_s42/summary.md)

---

## 1. classification 은 무엇인가

classification 은 입력을 미리 정해진 클래스 중 하나로 보내는 문제다.
이번 stage 는 이진 분류(binary classification) 문제를 다룬다.

- 입력 `x`: 사람의 속성(나이, 학력, 직업, 근로 시간 등)
- 출력 `y`: `<=50K` 또는 `>50K`

처음에는 “그냥 맞추면 되는 문제”처럼 보이지만, 실제로는 다음 세 단계를 구분해야 한다.

1. **feature representation** — 입력을 모델이 읽을 수 있게 정리한다.
2. **score estimation** — 모델이 양성일 가능성(score)을 계산한다.
3. **decision rule** — threshold 를 적용해 최종 label 을 만든다.

즉, 분류는 곧바로 label 이 튀어나오는 문제가 아니라 **score 를 만들고 decision 을 내리는 과정**이다.

---

## 2. 왜 이 이론이 필요한가

표형 분류는 매우 실용적이지만, 동시에 초보자가 가장 많이 헷갈리는 분야이기도 하다.

### 이유 1. accuracy 가 너무 쉽게 속인다
클래스 비율이 80:20 이면 다수 클래스를 찍기만 해도 accuracy 0.8 이 나온다.
그래서 “높은 정확도 = 좋은 모델”이라는 오해가 생긴다.

### 이유 2. score 와 label 이 다르다
같은 모델이라도 threshold 를 0.3 으로 두느냐 0.7 로 두느냐에 따라 precision / recall / F1 이 달라진다.

### 이유 3. 평균 점수는 failure pattern 을 숨긴다
전체 AUROC 가 높아도 특정 slice 에서만 반복적으로 틀릴 수 있다.

### 이유 4. 확률을 믿을 수 있는지는 별개의 문제다
ranking 이 좋다고 calibration 이 자동으로 좋아지는 것은 아니다.

그래서 Stage 1 은 “분류기 하나 돌리기”가 아니라, **분류 실험을 해석하는 언어를 만드는 단계**다.

---

## 3. score, threshold, label

### 3.1 score 란 무엇인가

score 는 모델이 “이 샘플이 양성일 가능성이 얼마나 큰가”를 표현한 연속값이다.

- logistic regression: 보통 sigmoid 확률
- random forest: tree vote 기반 확률
- MLP: softmax / sigmoid 확률

### 3.2 threshold 는 왜 중요한가

보통 score >= 0.5 이면 양성으로 예측하지만, 0.5 는 법칙이 아니라 **선택값**이다.

- threshold 를 높이면 precision 은 올라가기 쉽다.
- threshold 를 낮추면 recall 은 올라가기 쉽다.

즉, threshold 는 모델의 가치 판단을 바꾸는 장치다.

### 3.3 label 은 최종 의사결정이다

label 은 모델이 실제로 “양성 / 음성”을 선언한 결과다.
그래서 threshold 기반 metric(F1, accuracy)은 score quality 와는 다른 질문에 답한다.

---

## 4. 왜 class imbalance 가 문제인가

이번 Adult 실험에서는 `>50K` 가 소수 클래스다.
이 구조에서는 다수 클래스를 계속 찍기만 해도 accuracy 가 꽤 높게 나온다.

이 때문에 baseline 으로 `DummyClassifier` 를 꼭 넣는다.

- dummy 가 높으면 데이터 구조가 accuracy 를 부풀릴 수 있음을 뜻한다.
- 그때는 AUPRC 같은 metric 이 더 중요해진다.

즉, class imbalance 는 단순한 데이터 통계가 아니라 **어떤 metric 을 믿어야 하는지 바꾸는 요소**다.

---

## 5. Stage 1 메트릭을 자세히 이해하기

## 5.1 Accuracy

전체 예측 중 맞은 비율이다.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

가장 직관적이지만, 이번 stage 같은 불균형 데이터에서는 다수 클래스를 많이 맞히기만 해도 높아진다.

## 5.2 Precision

양성이라고 예측한 것 중 실제로 양성인 비율이다.

\[
Precision = \frac{TP}{TP + FP}
\]

“양성이라고 말할 때 얼마나 신중한가?”를 본다.

## 5.3 Recall

실제 양성 중에서 모델이 얼마나 잡았는지 본다.

\[
Recall = \frac{TP}{TP + FN}
\]

“놓친 positive 가 얼마나 많은가?”에 답한다.

## 5.4 F1

precision 과 recall 의 조화 평균이다.

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

양성 class 를 중요하게 볼 때 유용하지만, threshold 선택의 영향을 많이 받는다.

## 5.5 AUROC

threshold 전체를 훑으며 ranking quality 를 본다.
양성과 음성을 얼마나 잘 분리하는지 보는 metric 이다.

하지만 positive 가 희소할 때는 실제 운영 감각보다 낙관적으로 보일 수 있다.

## 5.6 AUPRC

precision-recall 곡선 아래 면적이다.
양성 class 가 드문 상황에서 **positive 를 얼마나 앞쪽으로 끌어올리는지**를 더 민감하게 본다.

그래서 이번 stage 에서는 accuracy 보다 AUPRC 를 더 중요하게 읽는다.

## 5.7 Calibration

“0.9 확률을 준 샘플이 실제로 90% 정도 맞는가?”를 확인한다.
운영 환경에서는 이 질문이 매우 중요하다. 모델이 틀리면서도 매우 자신 있게 말하면 위험하기 때문이다.

---

## 6. 전처리는 왜 pipeline 안에서 해야 하는가

Adult 데이터에는 수치형과 범주형이 섞여 있고, `?` 같은 missing-like token 도 있다.
그래서 전처리를 이렇게 분리한다.

- numeric: 결측 대체, 필요하면 scaling
- categorical: 결측 대체, one-hot encoding

여기서 가장 중요한 원칙은 **fit 은 train 에서만** 해야 한다는 점이다.

왜냐하면 validation / test 의 정보를 encoder 나 imputer 가 미리 보면 data leakage 가 생기기 때문이다.
그래서 `ColumnTransformer` 와 `Pipeline` 을 쓴다.

---

## 7. 이번 stage 에서 쓰는 모델들

### 7.1 DummyClassifier

아무 학습도 하지 않는 baseline 이다.
이 모델이 필요한 이유는 데이터 구조가 metric 을 얼마나 속일 수 있는지 보여 주기 때문이다.

### 7.2 Logistic Regression

선형 분류기다.
- 빠르다.
- 해석이 쉽다.
- one-hot tabular 에서 baseline 으로 매우 강하다.

선형 모델이 꽤 잘 나오면, 데이터가 완전히 복잡하기만 한 것은 아니라는 संकेत이다.

### 7.3 Random Forest

여러 decision tree 를 모아 예측하는 ensemble 이다.
- feature interaction 을 잘 잡는다.
- tabular strong baseline 으로 자주 쓰인다.
- 해석은 logistic 보다 어렵지만, 성능은 보통 더 강하다.

### 7.4 GPU MLP

다층 퍼셉트론 기반 신경망이다.
- GPU 사용 연습에 좋다.
- 비선형 표현력이 크다.
- 하지만 tabular 에서는 tree ensemble 보다 항상 낫지 않다.

Stage 1 에서 MLP 를 넣는 이유는 “딥러닝이니까 무조건 좋다”가 아니라, **tabular 에서 inductive bias 차이를 비교하기 위해서**다.

---

## 8. 데이터셋: Adult Census Income

이 데이터셋은 성인 개인의 인구통계 / 직업 정보를 기반으로 소득 구간을 예측한다.

왜 이 stage 에 특히 좋을까?

1. mixed-type tabular 이다.
2. class imbalance 가 있다.
3. slice analysis 를 하기에 좋은 속성이 있다.
4. high-confidence error 를 보면 모델이 사회경제적 archetype 을 어떻게 배우는지 드러난다.

즉, 이 데이터는 단순 benchmark 가 아니라 **분류 실험 해석 훈련용 교재**에 가깝다.

---

## 9. figure 를 읽는 방법

### class_distribution
baseline accuracy 가 왜 부풀려지는지 보여 준다.

### PR curve
희소 positive class 에서 가장 먼저 봐야 하는 곡선이다.

### ROC curve
threshold 전반 ranking 을 보여 준다. PR curve 와 함께 봐야 안전하다.

### confusion_matrix
FP / FN 구조를 본다. threshold 에서 어떤 실수가 더 많은지 직접 보게 해 준다.

### calibration_curve
confidence 를 믿을 수 있는지 확인한다.

### permutation_importance
모델이 어떤 feature 를 많이 활용했는지 본다.

### error_slice_by_sex
전체 평균이 가리는 demographic gap 을 본다.

### failure_examples
모델이 어떤 archetype 을 과신하는지 직접 보여 준다.

---

## 10. 이번 stage 에서 정말 공부해야 할 것

1. **metric 은 서로 다른 질문에 답한다**
2. **dummy baseline 은 해석의 출발점이다**
3. **threshold metric 과 ranking metric 을 분리해서 읽어야 한다**
4. **calibration 이 좋아야 confidence 를 운영에서 쓸 수 있다**
5. **failure slice 를 봐야 모델의 습관이 드러난다**

Stage 1 을 제대로 끝냈다는 뜻은, 단지 `random_forest` 가 이겼다고 말하는 것이 아니다.
다음 문장을 설명할 수 있어야 한다.

> 왜 이 데이터에서는 accuracy 보다 AUPRC 를 먼저 봐야 하고, 왜 high-confidence error 를 함께 읽어야 하며, 왜 tree ensemble 이 MLP 보다 더 안정적인 ranking 을 냈는가?

이 질문에 답할 수 있으면, Stage 1 은 이미 단순 실험이 아니라 **실제 ML 읽기 훈련**이 된다.
