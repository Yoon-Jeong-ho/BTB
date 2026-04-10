# 03 Domain Adaptive Pretraining 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 domain shift, continued pretraining, catastrophic forgetting, data selection, stopping concern을 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- DAPT는 objective를 바꾸는 실험이 아니라 **같은 pretraining objective를 유지한 채 데이터 분포를 다시 가중**하는 실험이다.
- pure-domain continued pretraining은 in-domain validation loss를 빠르게 낮출 수 있지만 general-domain retention loss를 함께 악화시킬 수 있다.
- replay mixture는 general corpus를 일부 다시 보여 주어 forgetting을 늦추지만, domain adaptation 속도를 낮출 수 있다.
- data selection은 문서 수보다 품질, 중복, contamination risk, 목표 분포 적합도를 먼저 봐야 한다.
- stopping은 training loss 최저점 하나가 아니라 in-domain gain, general regression guardrail, downstream probe를 동시에 보는 Pareto decision이다.

## 확인 질문
- domain shift가 vocabulary 차이보다 더 넓은 문체·형식·최신성 차이라는 점을 어떻게 관측할 수 있는가?
- pure domain schedule이 왜 adaptation speed와 catastrophic forgetting을 동시에 키울 수 있는가?
- replay mixture가 retention을 지키는 대신 specialization 속도를 늦추는 이유는 무엇인가?
- data selection에서 noisy large corpus보다 curated small corpus가 나을 수 있는 조건은 무엇인가?
- stop step을 정할 때 in-domain loss와 general retention loss가 충돌하면 어떤 기준을 우선할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): domain shift, continued pretraining, replay mixture, forgetting/stopping trade-off를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
