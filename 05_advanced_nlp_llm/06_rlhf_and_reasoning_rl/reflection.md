# 06 RLHF and Reasoning RL 학습자 회고

## 실행 전 예측
1. reward model이 truth engine이 아니라 preference proxy라면, 어떤 rubric 항목이 reward를 가장 쉽게 왜곡할지 예측해 보라.
2. RLHF에서 PPO-family policy update를 할 때 reward만 보지 않고 KL anchor를 보는 이유를 한 문장으로 적어 보라.
3. reasoning RL의 reward shaping에서 final answer reward와 process reward 중 어느 쪽이 verifier signal과 더 직접 연결될지 예상해 보라.

## 실행 후 관찰
1. `scratch_lab.py`의 chosen/rejected reward margin을 보고 reward model이 어떤 응답을 더 선호했는지 설명하라.
2. `rlhf_reasoning_reward.svg`에서 verifier bonus가 높은데 judge signal과 완전히 같은 뜻은 아닌 사례를 찾아 보라.
3. `framework_lab.py`에서 reward mean과 advantage mean이 오른 뒤에도 KL guardrail을 확인하는 이유를 써 보라.
4. answer accuracy와 verifier consistency가 모두 올라도 judge_length_bias_flag가 남는다면 어떤 추가 평가가 필요한가?

## failure mode 질문
1. reward hacking은 이 toy setup에서 어떤 방식으로 생길 수 있는가? verifier checklist만 맞추는 답과 실제 좋은 답을 구분해 보라.
2. verbosity가 늘어난 응답을 judge가 더 좋아한다면 reasoning RL은 어떤 잘못된 reward shaping을 배울 수 있는가?
3. safety reward가 과하면 over-refusal이 늘 수 있다. 어떤 prompt slice로 이를 감시할 것인가?
4. judge는 높게 평가했지만 verifier가 낮게 본 응답이 있다면, 그 불일치는 어떤 blind spot을 말하는가?

## 다음 단위로 넘길 메모
- RLHF / reasoning RL이 해결한 것: 내부 policy behavior를 reward model, verifier, judge signal로 더 직접 shaping한다.
- 아직 남은 것: reward model이 truth engine이 아니므로 retrieval grounding, factuality probes, held-out eval이 필요하다.
- 다음 RAG/eval 단위에서는 "좋아 보이는 답"과 "근거가 있는 답"을 분리해서 본다.
