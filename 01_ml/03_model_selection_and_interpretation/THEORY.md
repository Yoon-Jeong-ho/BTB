# 03 Model Selection And Interpretation 이론

## 왜 이 실습을 하는가

점수 한 번 잘 나온 모델보다, **재현 가능한 validation 전략과 선택 근거**가 더 중요하다.

## 핵심 이론

### 1. TimeSeriesSplit

시간 데이터에서 미래를 과거로 섞으면 leakage가 생긴다.
따라서 fold도 시간 순서를 지키는 방식이 필요하다.

### 2. Validation curve

하이퍼파라미터가 커질수록 항상 좋아지는 것이 아니다.
Validation curve는 과소적합/과적합의 균형을 본다.

### 3. Permutation importance

feature를 섞었을 때 점수가 얼마나 떨어지는지로 중요도를 본다.
모델 내부 계수/feature importance보다 더 모델-불문 해석에 가깝다.

## 분석 포인트

- 출퇴근 시간대/날씨/working day에서 오차가 어떻게 변하는가
- best score뿐 아니라 fold 분산이 안정적인가
- count 회귀에서 peak demand를 제대로 따라가는가
