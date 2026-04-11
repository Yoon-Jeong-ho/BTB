# 01 Paper Reproduction Playground 분석 프레임

이 파일은 실행할 때마다 덮어쓰지 않는 stable analysis markdown이다. 실제 run 관측은 `analysis.py`가 `artifacts/analysis-manual/latest_report.md`에 생성한다.

## 관련 이론
- [THEORY.md](./THEORY.md)
- [PREREQS.md](./PREREQS.md)

## 읽는 순서
1. `scratch_lab.py`가 만든 claim/evidence matrix를 먼저 본다. claim을 얼마나 좁혔고, evidence type과 acceptance rule이 claim에 맞는지 확인한다.
2. baseline/reported/reproduced comparison을 세 층으로 분리한다.
   - reported baseline vs reported method는 paper가 주장한 margin을 읽는 층이다.
   - reproduced baseline vs reproduced method는 같은 protocol 안에서 해석 가능한 primary comparison이다.
   - reported vs reproduced gap은 mismatch hypothesis를 세우는 힌트이지 즉시 성공/실패 판정이 아니다.
3. scope control을 확인한다. reduced claim, dataset proxy, fixed CPU budget이라면 absolute paper reproduction이라고 말하지 않는다.
4. variance를 본다. seed std가 reported margin이나 reproduced gap과 비슷하면 결론을 낮춘다.
5. mismatch hypothesis를 artifact로 남긴다. preprocessing_alignment, seed_variance, budget_mismatch, evaluator_mismatch를 다음 실험 체크리스트로 바꾼다.
6. artifact hygiene를 확인한다. scope boundary, comparison table, variance summary, mismatch hypotheses, manifest가 모두 남아야 다음 사람이 이어서 실험할 수 있다.

## 해석 원칙
- claim/evidence matrix는 “무엇을 믿고 싶은가”와 “무엇으로 확인했는가”를 강제로 붙이는 장치다.
- baseline은 paper 표에서 가져오는 숫자만으로 충분하지 않다. 같은 local protocol에서 reproduced baseline을 다시 세워야 method delta가 해석된다.
- reported result와 reproduced result의 차이는 흔히 preprocessing, evaluator, seed variance, budget mismatch에서 온다.
- mismatch hypothesis는 실패를 꾸미는 문장이 아니라 다음 run에서 어떤 변수를 잠글지 알려 주는 연구 로그다.
- artifact hygiene는 연구 재현성의 최소 단위다. 실행 파일보다 중요한 것은 다음 사람이 내 scope와 비교 조건을 오해하지 않게 하는 것이다.
