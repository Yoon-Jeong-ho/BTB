# 05 Preference Optimization: DPO, ORPO, KTO 이론 노트

## 핵심 개념

### 1. preference data는 정답지가 아니라 상대 선호 기록이다
chosen/rejected pair는 같은 prompt에 대해 한 응답이 다른 응답보다 더 낫다고 판단된 기록이다. chosen은 완벽한 정답이라는 뜻이 아니고, rejected도 항상 완전히 틀렸다는 뜻이 아니다. annotator 기준, rubric, policy version, safety policy가 바뀌면 선호 관계도 바뀔 수 있다.

그래서 preference optimization은 정답 복원보다 **policy가 어느 응답에 더 높은 log-prob를 주도록 밀 것인가** 에 가깝다. 이 단위의 scratch lab은 각 pair에서 `chosen_logprob - rejected_logprob` 형태의 log-prob margin을 계산한다. margin이 양수면 policy가 chosen 쪽을 더 선호하고, 음수면 아직 rejected 쪽이 더 쉽게 나오는 상태다.

### 2. policy update without full RL
전형적인 RLHF 설명은 SFT policy, reward model, online rollout, PPO류 update를 포함한다. DPO / ORPO / KTO는 이 heavy loop를 매번 요구하지 않고, offline preference dataset에서 policy log-prob를 직접 다루는 쪽으로 문제를 단순화한다.

중요한 감각은 다음과 같다.
- reward model 점수를 새로 학습하지 않아도 preference objective를 만들 수 있다.
- online environment rollout 없이도 저장된 chosen/rejected 또는 desirable/undesirable sample에서 update direction을 계산할 수 있다.
- 그래도 policy update이므로 drift, over-optimization, eval tradeoff를 봐야 한다.

즉 full RL을 쓰지 않는다는 말은 alignment 문제가 사라진다는 뜻이 아니라, **선호 신호를 supervised-like offline loss로 먼저 처리해 보는 것** 이다.

### 3. DPO: reference-relative chosen/rejected margin
DPO(Direct Preference Optimization)는 pairwise preference data에 잘 맞는다. high-level로는 current policy가 chosen을 rejected보다 더 높게 보도록 만들되, reference policy 대비 얼마나 달라졌는지를 같이 본다.

관찰값은 다음처럼 읽을 수 있다.
- `policy_margin = log p_policy(chosen) - log p_policy(rejected)`
- `reference_margin = log p_ref(chosen) - log p_ref(rejected)`
- `dpo_advantage = policy_margin - reference_margin`

DPO의 장점은 reward model 없이 chosen/rejected pair를 직접 objective로 사용할 수 있다는 점이다. 주의점은 reference policy와 beta/regularization 감각이 너무 강하면 변화가 약하고, 너무 약하면 style collapse나 factual drift가 생길 수 있다는 점이다.

### 4. ORPO: chosen likelihood anchor + odds-ratio preference term
ORPO(Odds Ratio Preference Optimization)는 chosen 응답을 잘 모방하는 SFT anchor와 chosen/rejected separation을 함께 보려는 관점으로 이해하면 좋다. reference model을 별도로 들고 가는 부담을 낮추면서, chosen answer likelihood를 유지하고 rejected보다 odds가 커지도록 신호를 준다.

교육적으로는 ORPO를 다음 질문으로 읽는다.
- chosen 응답 자체를 계속 잘 생성하게 붙잡고 있는가?
- chosen과 rejected의 odds-ratio가 벌어지는가?
- imitation anchor가 너무 강해서 preference separation이 약해지지는 않는가?

### 5. KTO: strict pair가 없어도 desirability label을 쓴다
KTO(Kahneman-Tversky Optimization)는 high-level에서 desirable / undesirable signal을 비대칭 utility처럼 다루는 preference optimization으로 볼 수 있다. DPO/ORPO처럼 strict chosen/rejected pair가 자연스러운 방법과 달리, KTO는 label-only setup에서도 사용할 수 있다는 유연성이 있다.

이 유연성은 annotation 비용을 낮출 수 있지만, label imbalance와 noise를 더 조심해야 한다. "desirable"이 무엇을 뜻하는지 rubric이 흔들리면 utility signal도 바로 흔들린다.

### 6. DPO / ORPO / KTO high-level contrast
| 방법 | 자연스러운 데이터 | anchor / regularization 직관 | 핵심 signal |
| --- | --- | --- | --- |
| DPO | chosen/rejected pair | reference policy 대비 drift 제어 | reference-relative chosen/rejected log-prob margin |
| ORPO | chosen/rejected pair | chosen likelihood anchor | chosen NLL + odds-ratio preference term |
| KTO | desirable/undesirable label | 구현별 utility anchor | desirable은 밀고 undesirable은 누르는 비대칭 utility |

세 방법 모두 reward model + online PPO 없이 preference를 직접 loss에 넣는다는 공통점이 있다. 그러나 데이터 요구사항, reference 처리, eval 해석은 다르다.

### 7. alignment/eval tradeoff
Preference objective가 좋아져도 alignment 품질이 자동으로 보장되지는 않는다. 특히 다음을 분리해서 봐야 한다.
- **length bias**: 더 긴 답이 judge에게 더 좋아 보이는가?
- **style over factuality**: 말투와 형식이 사실성보다 과대평가되는가?
- **over-refusal**: safety가 좋아진 것이 아니라 필요한 답도 과하게 거절하는가?
- **verbosity / latency**: preference score를 올리려다 비용과 지연이 커지는가?
- **distribution shift**: preference dataset 밖 prompt에서도 같은 개선이 유지되는가?

이 단위의 `analysis.py`는 stable `analysis.md`와 observed `latest_report.md`를 분리한다. stable file은 위 해석 프레임을 보존하고, observed report는 실행별 metrics를 바탕으로 DPO/ORPO/KTO와 alignment eval tradeoff를 다시 연결한다.

## Common Confusion
- chosen을 절대 정답으로 보는 실수
- rejected를 항상 harmful 또는 완전 오답으로 보는 실수
- DPO / ORPO / KTO를 이름만 다른 같은 loss로 보는 실수
- full RL을 쓰지 않으면 policy drift나 reward hacking 위험도 없어진다고 믿는 실수
- win rate 상승을 factuality와 safety 향상으로 바로 해석하는 실수
- KTO가 pair를 요구하지 않는다는 말을 label 품질이 덜 중요하다는 뜻으로 오해하는 실수

## 실행 결과 예시
```bash
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/scratch_lab.py
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/framework_lab.py
python 05_advanced_nlp_llm/05_preference_optimization_dpo_orpo_kto/analysis.py
```

실행 후에는 `artifacts/scratch-manual/preference_margin.svg`에서 policy/reference chosen-rejected margin을 보고, `artifacts/analysis-manual/latest_report.md`에서 DPO / ORPO / KTO contrast와 alignment eval tradeoff 해석을 확인한다.
