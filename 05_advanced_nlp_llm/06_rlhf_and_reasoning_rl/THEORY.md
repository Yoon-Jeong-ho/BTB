# 06 RLHF and Reasoning RL 이론 노트

## 핵심 개념
이 단위의 핵심은 RLHF와 reasoning RL을 거대한 학습 시스템이 아니라 **신호 설계와 policy update의 관찰 문제**로 읽는 것이다. reward model, verifier, judge, PPO-family update, reward shaping은 모두 모델 행동을 어느 방향으로 밀지 정하는 proxy이며, 이 proxy가 곧 진실이라는 뜻은 아니다.

### 1. reward model intuition: truth engine이 아니라 preference proxy
- reward model은 chosen/rejected pair, ranked response, rubric score, verifier / judge signal을 보고 응답 선호를 scalar로 압축한다.
- 이 값은 factual correctness 자체가 아니라 annotation 기준, style preference, safety framing, length bias가 섞인 preference proxy다.
- 따라서 reward가 높아졌다는 말은 "현재 proxy가 좋아하는 방향으로 policy update가 일어났다"는 뜻이지, 모든 factuality와 safety 문제가 해결됐다는 뜻이 아니다.
- reward model을 truth engine으로 오해하면 reward hacking, verbosity inflation, over-refusal 같은 failure mode를 늦게 발견한다.

### 2. PPO/RLHF high-level loop
RLHF loop는 보통 다음 구조로 읽는다.
1. prompt를 샘플링한다.
2. 현재 policy로 rollout을 생성한다.
3. reward model, verifier, judge, rule-based checks로 scalar reward와 auxiliary signal을 만든다.
4. PPO-family 혹은 advantage-style policy update로 높은 reward 방향을 강화한다.
5. KL anchor와 held-out eval로 drift, factuality, safety, verbosity regression을 확인한다.

여기서 PPO는 반드시 실제 대형 PPO 구현을 뜻하지 않는다. 이 단위에서는 "reward advantage를 키우되 policy가 기준 분포에서 너무 멀어지지 않게 보는 high-level framing"으로 사용한다.

### 3. reasoning RL과 reward shaping
reasoning RL은 최종 정답률만 보는 outcome reward와 과정 품질을 보는 process reward를 함께 생각한다.
- outcome reward: 최종 답이 맞았는가, 사용자 과업이 해결됐는가.
- process reward: 중간 계산이 verifier를 통과했는가, tool result와 final answer가 일치하는가, self-correction이 일어났는가.
- hybrid reward shaping: outcome, verifier, judge, format, safety, KL penalty를 섞어 policy update 방향을 만든다.

중요한 오해는 "긴 chain-of-thought가 곧 좋은 reasoning"이라는 생각이다. 긴 trace는 judge를 설득하는 표면 신호가 될 수 있지만, verifier consistency나 final answer accuracy가 낮으면 reasoning RL이 아니라 verbosity optimization에 가깝다.

### 4. verifier와 judge signal
- verifier는 좁고 구조화된 signal이다. 수식, 형식, tool-result consistency처럼 체크리스트화하기 쉽다.
- judge는 넓고 비교적인 signal이다. helpfulness, clarity, safety tone, user preference를 한 번에 비교할 수 있다.
- verifier는 검사하지 않는 오류를 놓치고, judge는 length bias와 style bias에 흔들린다.
- 둘이 모두 높아도 reward hacking 가능성은 남는다. 둘이 불일치하면 해당 prompt slice를 따로 분석해야 한다.

### 5. failure modes
- reward hacking: policy가 실제 품질보다 reward model의 약점을 공략한다.
- verbosity inflation: judge가 긴 설명을 선호해 불필요한 trace가 늘어난다.
- over-refusal: safety reward가 과해져 답할 수 있는 요청도 거절한다.
- format gaming: verifier checklist를 통과하는 형식만 맞추고 내용은 약해진다.
- style over factuality: 설득력 있는 문체가 사실 정확도를 가린다.

## 실행 결과 예시와 해석
`scratch_lab.py`는 reward model batch에서 chosen reward가 rejected reward보다 높은지 보고, `rlhf_reasoning_reward.svg`로 chosen reward / rejected reward / verifier bonus를 비교한다. `framework_lab.py`는 reward mean과 advantage mean이 올라가도 KL guardrail 안에 남는지, answer accuracy와 verifier consistency가 함께 좋아지는지 확인한다.

핵심 질문은 숫자가 큰가가 아니라 다음이다.
- 이 reward model은 무엇의 proxy인가?
- policy update가 어떤 behavior를 강화했는가?
- verifier와 judge가 놓칠 수 있는 blind spot은 무엇인가?
- reasoning RL reward shaping이 trace length가 아니라 검증 가능성과 정답성을 밀고 있는가?
- reward hacking, verbosity, over-refusal을 감시하는 eval slice가 있는가?
