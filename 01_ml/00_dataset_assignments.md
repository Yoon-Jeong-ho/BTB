# 00 ML Dataset Assignments

이 문서는 `BTB` 프로젝트에서 ML 트랙을 진행할 때 실제로 사용할 `확정 데이터셋` 을 고정한다.

원칙은 단순하다.

1. 각 step마다 `Primary dataset` 은 하나만 둔다.
2. 가능하면 Hugging Face `datasets` 또는 바로 접근 가능한 공개 소스를 쓴다.
3. 실습 난이도는 단계적으로 올라가야 한다.

## 확정 데이터셋 표

| Step | Primary dataset | 접근 방식 | 왜 이 단계에 맞는가 |
| --- | --- | --- | --- |
| 01 | `scikit-learn/adult-census-income` | Hugging Face `datasets.load_dataset(...)` | 범주형/수치형 혼합 분류, 결측/전처리, threshold metric 연습에 가장 좋다 |
| 02 | `California Housing` | `sklearn.datasets.fetch_california_housing(as_frame=True)` | 회귀, residual, outlier, parity plot 연습에 가장 빠르다 |
| 03 | `Bike Sharing Dataset` | UCI 공개 데이터셋 | time-aware validation, leakage 방지, count regression을 배울 수 있다 |
| 04 | `mstz/covertype` | Hugging Face `datasets.load_dataset(...)` | 서버 학습 전 대형 tabular multiclass benchmark로 적당하다 |

## Step별 로딩 기준

### Step 01

```python
from datasets import load_dataset

ds = load_dataset("scikit-learn/adult-census-income", split="train")
df = ds.to_pandas()
```

### Step 02

```python
from sklearn.datasets import fetch_california_housing

frame = fetch_california_housing(as_frame=True)
df = frame.frame
```

### Step 03

```python
from ucimlrepo import fetch_ucirepo

bike = fetch_ucirepo(id=275)
X = bike.data.features
y = bike.data.targets
```

### Step 04

```python
from datasets import load_dataset

ds = load_dataset("mstz/covertype", split="train")
df = ds.to_pandas()
```

## 서버 확장용 선택 데이터셋

`HIGGS` 는 step 04 이후의 추가 확장으로 둔다. 이유는 단순하다.

- 접근은 쉽지만 용량이 커서 로컬 실습용으로는 과하다.
- `Covertype` 로 실험 규약을 먼저 굳힌 뒤 서버에서 넘어가는 편이 낫다.

공식 출처:

- HF Adult: https://huggingface.co/datasets/scikit-learn/adult-census-income
- sklearn California Housing: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- UCI Bike Sharing: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- HF Covertype: https://huggingface.co/datasets/mstz/covertype
- UCI HIGGS: https://archive.ics.uci.edu/dataset/280/higgs
