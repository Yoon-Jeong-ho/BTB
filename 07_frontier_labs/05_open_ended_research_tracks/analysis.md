# 05 Open-Ended Research Tracks 분석

## Stable interpretation

Open-ended research is not scope-less exploration. It is an operating system for turning a north-star question into a bounded research scope, a hypothesis registry, an iteration boundary, kill criteria, an evidence standard, and a decision note. This unit keeps the simulation CPU-safe and deterministic so that the research-operations reasoning is testable without any external service, live model, or GPU.

## Korean-first reading

- research scope는 큰 north-star question을 이번 track에서 실제로 좁힐 질문으로 자르는 문서다. out-of-scope를 먼저 적어야 scope creep를 줄일 수 있다.
- hypothesis registry는 아이디어 목록이 아니라 claim, mechanism guess, iteration boundary, kill criteria, evidence standard, reopen condition을 묶은 운영 표다.
- iteration boundary는 무엇을 바꾸고 무엇을 고정하며 몇 번까지 retry할지 정한다. boundary가 없으면 결과 해석은 retrospective story-telling이 되기 쉽다.
- evidence standard는 exploratory phase에서도 느슨해지면 안 된다. baseline-relative signal, failure slice notes, qualitative examples, negative result log, inconclusive reason을 같이 남긴다.
- negative result와 inconclusive result는 다른 판단이다. negative result는 충분한 관찰이 현재 가설을 지지하지 않는 것이고, inconclusive result는 measurement나 boundary가 약해서 아직 결론을 못 내리는 것이다.
- stop / pause / escalate / archive decision은 연구 열정을 꺾는 장치가 아니라, 다음 iteration이 같은 실패를 비싸게 반복하지 않게 하는 운영 장치다.
- archive note에는 성공한 것뿐 아니라 중단 이유와 reopen condition도 들어가야 한다.

## Observed run

`analysis.py`는 `artifacts/scratch-manual/metrics.json`과 `artifacts/framework-manual/metrics.json`을 읽어 실행별 관측 보고서 `artifacts/analysis-manual/latest_report.md`를 쓴다. 이 문서는 stable report이며 실행별 숫자와 decision log는 generated artifact에서 확인한다.

## 관련 이론

- [THEORY.md](./THEORY.md) — open-ended research 운영 원리
- [PREREQS.md](./PREREQS.md) — 필요한 선행 감각
- 이전 단위 [04_benchmark_and_dataset_construction](../04_benchmark_and_dataset_construction/README.md) — evidence trust와 benchmark contract 감각
