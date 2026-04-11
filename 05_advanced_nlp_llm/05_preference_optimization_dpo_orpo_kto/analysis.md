# 05 Preference Optimization: DPO, ORPO, KTO 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 chosen/rejected pair, log-prob margin, DPO/ORPO/KTO contrast, policy update without full RL, alignment/eval tradeoff를 읽는 **안정적인 프레임**만 남겨 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- preference data의 chosen 응답은 절대 정답이 아니라 rejected보다 선호된 응답이다.
- log-prob margin은 같은 prompt에서 policy가 chosen 쪽에 얼마나 더 높은 확률을 주는지 보는 최소 관찰값이다.
- DPO는 reference-relative chosen/rejected margin을 직접 키우는 pairwise objective로 읽는다.
- ORPO는 chosen likelihood anchor와 odds-ratio preference term을 함께 보는 one-stage preference objective로 읽는다.
- KTO는 strict pair가 없어도 desirable/undesirable label을 비대칭 utility처럼 사용할 수 있다는 점을 강조한다.
- full RL loop 없이도 offline preference objective로 policy를 움직일 수 있지만, eval은 win rate 하나가 아니라 factuality, refusal balance, verbosity, style bias를 나눠 봐야 한다.

## 확인 질문
- chosen/rejected pair가 정답/오답과 다르다는 사실이 loss 해석을 어떻게 바꾸는가?
- reference-relative margin을 쓰면 policy drift는 줄지만 어떤 보수성이 생길 수 있는가?
- ORPO의 chosen likelihood anchor는 imitation과 preference separation 사이에서 어떤 균형을 만든다고 볼 수 있는가?
- KTO처럼 label-only signal을 쓰면 pairwise ranking보다 무엇이 유연해지고 무엇이 약해지는가?
- offline alignment eval에서 length bias, style over factuality, over-refusal을 어떻게 따로 감시할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): DPO / ORPO / KTO의 데이터 요구사항, anchor, alignment trade-off를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
