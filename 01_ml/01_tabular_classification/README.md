# 01 Tabular Classification

이 stage 는 “분류 모델을 하나 돌려 보기”가 아니라, **분류 문제를 읽는 언어를 배우는 단계**다.
특히 표형 데이터에서 자주 나오는 질문들을 실제 실험과 연결해 이해하도록 구성한다.

- classification 이란 무엇인가?
- 왜 score 와 label 을 구분해야 하는가?
- 왜 accuracy 하나만 보면 안 되는가?
- AUPRC, AUROC, F1, calibration 은 각각 어떤 질문에 답하는가?
- 어떤 모델이 어떤 데이터 구조에서 유리한가?
- 평균 점수 뒤에 숨어 있는 failure slice 는 어떻게 읽는가?

---

## 먼저 어디서부터 읽으면 좋은가

1. [THEORY.md](THEORY.md) — 용어, 개념, 메트릭, 모델 직관을 먼저 잡는다.
2. [최신 artifact 리포트](artifacts/20260327-164446_adult-census-income_model-suite_s42/README.md) — 이론이 실제 숫자와 figure 에서 어떻게 드러나는지 본다.
3. [최신 artifact 요약](artifacts/20260327-164446_adult-census-income_model-suite_s42/summary.md) — 핵심만 다시 압축해 본다.
4. 코드가 궁금하면 아래 stage-local Python 파일을 따라간다.

---

## classification 이란 무엇인가

classification 은 입력 `x` 를 미리 정해진 범주(label) 중 하나로 배정하는 문제다.
이번 stage 에서는 `Adult Census Income` 데이터로 다음 이진 분류를 다룬다.

- 입력: 나이, 학력, 직업, 결혼 상태, 근로 시간, 자본 이득/손실 등
- 출력: `<=50K` 또는 `>50K`

여기서 중요한 것은 모델이 보통 **바로 label 을 내는 것이 아니라 score / probability 를 먼저 낸다**는 점이다.

- score: “양성일 가능성”에 대한 연속값
- threshold: score 를 label 로 바꾸는 기준
- label: 최종 분류 결과

그래서 같은 모델이어도 threshold 를 바꾸면 precision, recall, F1 이 바뀐다.

---

## 이 stage 에서 자주 나오는 용어

| 용어 | 뜻 | 이번 stage 에서 왜 중요한가 |
| --- | --- | --- |
| class imbalance | 클래스 비율이 기울어진 상태 | dummy baseline accuracy 가 높아 보이는 착시를 만든다 |
| score / probability | 모델이 내는 연속값 | label 이전의 ranking 품질을 본다 |
| threshold | score 를 label 로 바꾸는 기준 | F1, precision, recall 이 여기서 달라진다 |
| calibration | confidence 와 실제 정답률의 일치 정도 | 고확신 오답을 얼마나 믿어야 하는지 결정한다 |
| slice analysis | 특정 그룹별 성능 비교 | 전체 평균이 가리는 실패 패턴을 드러낸다 |
| high-confidence error | 높은 score 로 틀린 샘플 | 모델이 무엇을 과신하는지 보여 준다 |

---

## 이 stage 에서 쓰는 메트릭

### Accuracy
전체 예측 중 맞은 비율이다. 직관적이지만 불균형 데이터에서는 쉽게 높아 보인다.

### Precision / Recall / F1
- Precision: 양성이라고 한 것 중 실제 양성 비율
- Recall: 실제 양성 중 얼마나 잡았는지
- F1: precision 과 recall 의 균형

### AUROC
threshold 전반에서 score ranking 이 얼마나 좋은지 본다. 다만 positive 가 희소할 때는 낙관적으로 보일 수 있다.

### AUPRC
희소한 positive class 를 얼마나 앞쪽에 끌어올리는지 보여 준다.
이번 stage 에서는 **가장 중요하게 보는 메트릭**이다.

### Calibration
0.9 확률을 준 샘플이 실제로 90%쯤 맞는지 본다. 운영 관점에서 confidence 를 믿을 수 있는지를 판단한다.

---

## 이번 stage 의 모델들: 무엇을 왜 쓰는가

| 모델 | 성격 | 어디에 쓰이는가 | 이번 실험에서 보는 포인트 |
| --- | --- | --- | --- |
| `DummyClassifier` | 아무것도 배우지 않는 baseline | 최소 기준선 확인 | accuracy 착시를 보여 준다 |
| `LogisticRegression` | 선형 baseline | 빠르고 해석 가능한 기본 분류기 | 전처리가 좋으면 생각보다 강하다는 점 |
| `RandomForestClassifier` | 비선형 tree ensemble | tabular strong baseline | feature interaction 을 잘 잡는지 |
| `GPU MLP` | 딥러닝 비교군 | GPU 사용, 비선형 표현 | tabular 에서 항상 tree 보다 좋지 않다는 점 |

이 조합을 쓰는 이유는 “최고 성능 모델 찾기”보다 **baseline → strong baseline → neural comparator** 흐름을 공부하기 좋기 때문이다.

---

## 데이터셋은 무엇이고 왜 이걸 쓰는가

이번 stage 의 기준 데이터셋은 `scikit-learn/adult-census-income` 이다.

```python
from datasets import load_dataset

ds = load_dataset('scikit-learn/adult-census-income', split='train')
df = ds.to_pandas()
```

이 데이터가 좋은 이유는 다음과 같다.

- 수치형 / 범주형 feature 가 함께 있다.
- `?` 로 표시된 missing-like token 이 있다.
- 양성 클래스(`>50K`)가 상대적으로 적다.
- 성별, 학력, 직업 같은 slice 해석이 가능하다.
- 고확신 오답을 읽으면 모델이 가진 사회경제적 archetype 이 드러난다.

즉, 단순한 toy example 이 아니라 **표형 분류 실험의 핵심 함정이 압축된 데이터**다.

---

## 이번 실험에서 특히 주의 깊게 볼 것

1. **baseline accuracy 에 속지 말 것**
   - class distribution 을 먼저 본다.
2. **AUPRC 와 AUROC 를 함께 읽을 것**
   - ranking 품질과 positive 탐지 품질은 다르다.
3. **threshold metric 과 ranking metric 을 구분할 것**
   - F1 과 AUPRC 는 같은 질문에 답하지 않는다.
4. **calibration 을 볼 것**
   - score 를 운영에서 믿을 수 있는지 확인한다.
5. **failure slice 를 볼 것**
   - sex, education, occupation 조합에서 반복되는 오류를 확인한다.

---

## figure 를 읽는 순서

### 1) 결과 figure

- [class_distribution.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/results/class_distribution.svg)
- [pr_curve.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/results/pr_curve.svg)
- [roc_curve.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/results/roc_curve.svg)
- [confusion_matrix.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/results/confusion_matrix.svg)
- [calibration_curve.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/results/calibration_curve.svg)

읽는 순서의 이유는 간단하다.

1. 데이터 구조를 먼저 보고
2. ranking 품질을 보고
3. threshold 결과를 보고
4. confidence 신뢰도를 본다.

### 2) 분석 figure

- [permutation_importance.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/analysis/permutation_importance.svg)
- [error_slice_by_sex.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/analysis/error_slice_by_sex.svg)
- [confidence_vs_correctness.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/analysis/confidence_vs_correctness.svg)
- [failure_examples.svg](artifacts/20260327-164446_adult-census-income_model-suite_s42/figures/analysis/failure_examples.svg)

여기서는 “왜 그런 결과가 나왔는가”를 본다.

---

## stage-local 코드 구조

이제 Stage 1 코드는 이 폴더 안에서 바로 읽히도록 쪼개 두었다.

- `dataset.py` — Adult 데이터 로드, split, 전처리 준비
- `models.py` — baseline / strong baseline / GPU MLP 학습 로직
- `analysis.py` — prediction sample, result figure, analysis figure 생성
- `report.py` — artifact README / summary 생성
- `experiment.py` — Stage 1 전체 실행 orchestration
- `run_stage.py` — CLI entrypoint

실행 예시는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/01_tabular_classification/run_stage.py --gpu 0
```

---

## 최신 결과를 읽을 때 던질 질문

1. 왜 `random_forest` 가 AUPRC 기준 최고였는가?
2. 왜 `gpu_mlp` 는 accuracy 는 더 높은데 ranking 은 덜 안정적인가?
3. 고확신 오답은 어떤 속성 조합에 몰렸는가?
4. slice error 는 전체 평균이 놓치는 무엇을 보여 주는가?
5. 다음 단계에서 threshold tuning / calibration / boosting 중 무엇을 먼저 시도해야 하는가?

이 질문에 답할 수 있으면, Stage 1 은 “실험을 돌렸다”가 아니라 **분류 실험을 읽을 수 있게 되었다**고 볼 수 있다.
