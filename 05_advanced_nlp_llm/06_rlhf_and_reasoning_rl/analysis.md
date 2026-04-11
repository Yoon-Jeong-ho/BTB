# 06 RLHF and Reasoning RL 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy RLHF / reasoning RL 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 reward model intuition, PPO/RLHF high-level loop, verifier/judge signal, reasoning-oriented reward shaping, failure modes를 읽는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- reward model은 truth engine이 아니라 annotation rubric, verifier, judge가 압축된 preference proxy다.
- RLHF loop는 prompt sampling → policy rollout → reward scoring → PPO-family policy update → regression eval의 피드백 경로로 읽는다.
- policy update를 볼 때 reward mean만 보지 말고 KL anchor, reference drift, held-out safety/factuality slice를 같이 본다.
- reasoning RL은 긴 trace를 무조건 보상하는 것이 아니라 outcome reward와 process reward를 섞어 검증 가능성, self-correction, final answer quality를 함께 shaping한다.
- verifier는 좁고 체크리스트적인 signal을, judge는 넓고 비교적인 signal을 주지만 둘 다 reward hacking, length bias, over-refusal에 취약하다.

## 확인 질문
- reward model이 높은 점수를 준 응답은 어떤 rubric proxy에 맞았는가, 그리고 어떤 truth/factuality 축은 놓칠 수 있는가?
- PPO-family update sketch에서 reward가 오르더라도 KL guardrail을 같이 보는 이유는 무엇인가?
- verifier pass rate와 judge win rate가 서로 불일치하면 어떤 failure slice를 먼저 조사해야 하는가?
- reasoning-oriented reward shaping에서 trace length가 아니라 verifier consistency와 answer accuracy를 같이 보는 이유는 무엇인가?
- reward hacking, verbosity inflation, over-refusal을 관찰하려면 어떤 held-out regression prompts를 따로 유지해야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): reward model, RLHF, PPO-family update, verifier/judge, reasoning RL failure mode를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
