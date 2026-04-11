# 08 Alignment, Safety, and Model Behavior 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy alignment / safety behavior eval 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 alignment vs capability, refusal vs over-refusal, harmlessness / robustness, behavioral eval slice analysis, policy vs system-level safety를 읽는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- capability는 모델이 할 수 있는 일을 말하고, alignment는 그 능력이 정책 경계 안에서 어떤 행동으로 나타나는지를 말한다.
- refusal은 harmful request에서 필요하지만, benign request를 막으면 over-refusal이 되어 usefulness를 깎는다.
- harmlessness는 안전한 거절뿐 아니라 safe alternative, 범위 제한, 불확실성 표현을 포함한다.
- robustness는 paraphrase, noisy prompt, jailbreak-style phrasing에도 행동 계약이 유지되는지를 본다.
- behavioral eval은 하나의 scalar가 아니라 benign / harmful / borderline / robustness slice analysis로 읽어야 한다.
- policy vs system-level safety를 분리해야 model policy로 해결할 문제와 tool permission gating, moderation, audit logging으로 해결할 문제를 혼동하지 않는다.

## 확인 질문
- capability score가 높은데 behavior_contract_score가 낮다면 어떤 unsafe compliance나 policy drift가 숨어 있는가?
- harmful refusal rate가 높을 때 benign over-refusal rate도 같이 확인했는가?
- borderline request에서 safe alternative가 아니라 과잉 거절이나 우회 도움을 주고 있지는 않은가?
- robustness probe의 paraphrase / noisy / jailbreak variant 중 어느 표현이 behavior를 가장 많이 흔드는가?
- model policy 책임과 system guardrail 책임을 분리하지 않으면 어떤 product safety failure가 남는가?

## 관련 이론
- [THEORY.md](./THEORY.md): alignment vs capability, refusal / over-refusal, harmlessness, robustness, behavioral eval, system-level safety를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
