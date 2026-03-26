# 02 Tabular Regression 이론

## 왜 이 실습을 하는가

회귀는 단순히 점수 하나가 아니라, **오차가 어디에 남는지** 보는 습관이 중요하다.

## 핵심 이론

### 1. MAE vs RMSE

- MAE: 평균 절대 오차, 해석이 쉽다
- RMSE: 큰 오차에 더 민감하다

둘을 같이 보면, 모델이 "평균적으로 무난한지"와 "큰 실수를 자주 하는지"를 분리해 볼 수 있다.

### 2. Residual

`residual = prediction - target`

- residual이 0 근처에 모이면 좋다
- 특정 target 구간에서 residual이 한쪽으로 치우치면 systematic bias가 있다는 뜻이다

### 3. Parity plot

예측값이 실제값과 같다면 대각선 위에 놓인다.
대각선에서 멀어질수록 오차가 크다.

## 분석 포인트

- 고가 주택을 과소추정하는가?
- 특정 지역(lat/lon)에서 residual이 커지는가?
- 선형 모델과 tree 모델의 bias 차이가 어떻게 다른가?
