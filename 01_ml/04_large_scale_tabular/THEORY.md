# 04 Large Scale Tabular 이론

## 왜 이 실습을 하는가

대규모 tabular에서는 "정확도 최고"만이 아니라 **성능-시간-메모리 trade-off** 를 같이 봐야 한다.

## 핵심 이론

### 1. Accuracy vs Macro-F1

multiclass에서는 accuracy가 높아도 소수 class를 못 맞출 수 있다.
그래서 class별 균형을 보려면 macro-F1, macro-recall이 중요하다.

### 2. Throughput / Memory

서버 학습으로 넘어가면 fit time, predict time, peak memory도 모델 선택 기준이다.

### 3. GPU tree boosting

대형 tabular에서는 GPU histogram boosting/XGBoost가 실용적인 strong baseline이 될 수 있다.
다만 전처리/입출력 위치가 CPU/GPU 중 어디인지까지 같이 봐야 한다.

## 분석 포인트

- 어떤 class의 recall이 낮은가
- 어떤 class pair가 자주 헷갈리는가
- 더 많은 데이터를 넣을 때 성능 상승 폭이 유지되는가
- 성능 향상이 비용 증가를 정당화하는가
