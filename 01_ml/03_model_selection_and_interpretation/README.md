# 03 Model Selection And Interpretation

## 목표

하나의 점수보다 `재현성`, `모델 선택 근거`, `해석 가능성` 을 우선하는 실험 습관을 만든다.

## 이번 프로젝트 기준 확정 데이터셋

- 실행 코드: `run_stage.py`
- 이론 문서: [THEORY.md](THEORY.md)

- Primary: `Bike Sharing Dataset`
- Source: UCI Machine Learning Repository
- Load:

```python
from ucimlrepo import fetch_ucirepo

bike = fetch_ucirepo(id=275)
X = bike.data.features
y = bike.data.targets
```

- 이유: 같은 tabular라도 시간 축이 들어오면 split 전략과 모델 선택이 달라진다는 점을 배우기 좋다.
- Focus: `TimeSeriesSplit`, validation strategy, tuning cost, slice analysis

## 실습 파이프라인

1. 시간 정보를 유지하는 split 전략 설계
2. metric 우선순위 결정
3. 하이퍼파라미터 탐색 범위 문서화
4. best score뿐 아니라 분산까지 기록
5. permutation importance 또는 SHAP으로 해석
6. subgroup / slice 기반 에러 분석

## 결과로 남길 figure

- CV fold score boxplot
- validation curve
- top-k feature importance

## 분석으로 남길 figure

- subgroup metric comparison
- confidence bin plot
- common failure slice summary

## 승격 기준

- 최고 점수뿐 아니라 분산과 위험도까지 설명 가능하다.
- 다음 실험이 어떤 가설을 검증해야 하는지 명확하다.
