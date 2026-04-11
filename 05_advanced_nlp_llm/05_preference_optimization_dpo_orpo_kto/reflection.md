# 05 Preference Optimization 학습자 회고

## 실행 전 예측
1. chosen/rejected pair를 보면 어떤 응답이 "정답"인지보다 어떤 응답이 **상대적으로 더 선호**되었는지 먼저 표시해 보라.
2. `log-prob margin`이 음수인 pair가 하나라도 있으면, 그 pair에서 policy는 어떤 rejected 행동을 아직 더 쉽게 내는가?
3. DPO, ORPO, KTO 중 어떤 방법이 strict pair annotation에 가장 의존한다고 예상하는가?

## 실행 후 관찰
1. `scratch_lab.py`의 평균 policy margin과 DPO advantage를 비교하라. reference policy를 빼고 보면 어떤 해석이 빠지는가?
2. `framework_lab.py`에서 pair accuracy가 오른 뒤에도 reference drift guardrail을 보는 이유를 설명하라.
3. `preference_margin.svg`에서 policy margin과 reference margin이 가장 다르게 보이는 pair를 고르고, 그 차이가 어떤 behavior shift를 뜻하는지 써 보라.
4. KTO label examples에서 desirable/undesirable label만 있을 때 pairwise ordering 정보가 사라지는 대신 무엇이 쉬워지는지 정리하라.

## alignment/eval tradeoff 질문
1. win rate가 올랐지만 답변 길이도 늘었다면 length bias와 true helpfulness를 어떻게 분리할 것인가?
2. safety 관련 chosen 응답을 많이 밀었더니 over-refusal이 늘어났다면, 어떤 held-out prompt set을 추가해야 하는가?
3. style over factuality가 의심될 때 judge score 외에 어떤 factuality probe를 함께 봐야 하는가?
4. full RL loop 없이 offline preference objective만 쓴 이번 toy setup이 실제 post-training pipeline에서 충분하지 않을 수 있는 지점은 무엇인가?

## 다음 단위로 넘길 메모
- DPO / ORPO / KTO가 해결한 것: offline preference data를 직접 loss로 바꿔 policy update direction을 만든다.
- 아직 남은 것: online rollout, reward hacking, long-horizon reasoning signal, distribution shift, human eval calibration.
- 다음 RLHF 단위에서는 "왜 다시 reward model과 rollout이 필요한가?"라는 질문으로 이어 간다.
