# 04 Benchmark and Dataset Construction 분석

## 이 문서를 어떻게 읽을까
- 실행별 toy benchmark 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 benchmark/dataset construction을 해석하는 안정적인 프레임만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- benchmark는 leaderboard가 아니라 **task contract와 claim boundary를 고정하는 측정 계약**이다.
- dataset schema는 필드 목록만이 아니라 unit of record, source boundary, license tier, version freeze를 함께 포함해야 한다.
- source/split manifest는 random split보다 강한 hygiene를 요구한다. source와 template family가 split 사이를 건너면 leakage 위험이 커진다.
- annotation rubric과 QC는 label을 깨끗하게 보이게 하는 절차가 아니라 ambiguity와 disagreement를 기록하는 절차다.
- leakage, contamination, drift audit는 점수가 올랐을 때 그 점수를 capability improvement로 읽어도 되는지 확인하는 방어막이다.
- benchmark card, versioning, report template는 나중 연구 트랙이 숫자와 known limits를 함께 보고하게 만드는 운영 인터페이스다.

## 확인 질문
- task contract가 input/output/unit of record와 claim boundary를 명확히 고정하는가?
- dataset schema와 source/split manifest가 license, source, template family leakage를 같이 막는가?
- annotation rubric과 QC report가 agreement score뿐 아니라 major disagreement와 adjudication rule을 남기는가?
- contamination과 drift flag가 headline score 해석에 어떤 warning을 붙이는가?
- versioning 정책이 frozen core와 refresh slice를 구분해 과거 run과의 비교 가능성을 지키는가?

## 관련 이론
- [THEORY.md](./THEORY.md): benchmark card, task contract, dataset schema, source/split manifest, annotation rubric/QC, leakage/contamination/drift audit를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
