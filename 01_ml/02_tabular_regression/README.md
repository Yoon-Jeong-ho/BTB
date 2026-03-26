# 02 Tabular Regression

## 목표

회귀 문제에서 `loss`, `target scale`, `residual`, `outlier` 를 해석하는 습관을 만든다.

## 이번 프로젝트 기준 확정 데이터셋

- Primary: `California Housing`
- Source: scikit-learn builtin dataset
- Load:

```python
from sklearn.datasets import fetch_california_housing

frame = fetch_california_housing(as_frame=True)
df = frame.frame
```

- 이유: 설치 직후 바로 불러올 수 있고, 회귀 metric과 residual 분석을 빠르게 반복하기 좋다.
- Extension: `Ames Housing`

## 실습 파이프라인

1. target 분포와 이상치 확인
2. baseline으로 `DummyRegressor`, `LinearRegression`, `Ridge`
3. tree 계열 회귀 모델 추가
4. 먼저 random split 기반 회귀 실험을 고정
5. MAE, RMSE, R2 비교
6. residual을 구간별로 분석
7. feature importance와 error slice 분석

## 결과로 남길 figure

- target histogram
- learning curve
- parity plot
- residual histogram
- residual vs target scatter

## 분석으로 남길 figure

- feature importance
- worst prediction case table
- 지역/구간별 error slice plot

## 승격 기준

- residual 구조가 이해된다.
- target 구간별 성능 차이를 설명할 수 있다.
- 단순 평균 예측보다 충분히 나아진다.
