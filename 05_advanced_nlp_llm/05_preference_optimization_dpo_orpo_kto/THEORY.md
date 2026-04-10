# 05 Preference Optimization: DPO, ORPO, KTO 이론 노트

## 핵심 개념

### 1. preference data intuition: 정답 하나보다 "더 나은 응답" 신호를 다룬다
- preference data는 보통 절대 정답(answer key)보다 **상대 비교(relative preference)** 에 가깝다.
- 같은 프롬프트에 대해 다음 같은 형태가 등장한다.
  - chosen / rejected pair
  - ranked list
  - desirable / undesirable label
  - rule-based or model-based preference score
- 여기서 중요한 점은 chosen이 "완벽한 정답"이라는 뜻은 아니라는 것이다.
  - rejected보다 낫다는 뜻일 뿐, 여전히 사실 오류·장황함·과잉 거절을 포함할 수 있다.
  - annotator, policy version, rubric이 바뀌면 chosen/rejected 관계도 흔들릴 수 있다.
- 그래서 preference optimization은 정답 복원보다 **어느 방향으로 policy를 조금 더 밀고 덜 밀 것인가** 를 배우는 문제에 가깝다.

### 2. policy update without full RL: reward model + PPO를 꼭 거치지 않아도 된다
- 전통적인 RLHF 설명에서는 대체로 다음 루프가 나온다.
  1. SFT policy 준비
  2. preference data로 reward model 학습
  3. online rollout + PPO류 정책 최적화
- 하지만 post-training의 모든 선호 반영이 이 full RL framing을 꼭 필요로 하지는 않는다.
- DPO / ORPO / KTO 류는 **정책의 log-prob 자체를 직접 목적함수에 넣어**, reward model과 online sampling loop를 생략하거나 축소하는 방향을 취한다.
- 직관적으로는 다음 질문을 직접 loss로 쓰는 셈이다.
  - chosen 응답의 확률을 rejected보다 더 높일 것인가?
  - desirable 응답은 더 밀고 undesirable 응답은 덜 밀 것인가?
  - reference policy에서 너무 멀어지지 않도록 어느 정도 제동을 걸 것인가?
- 즉 "policy update"라는 말은 RL 용어를 빌리지만, 실제 구현 감각은 **오프라인 preference dataset 위에서 직접 log-prob 차이를 조절하는 supervised-like optimization** 에 더 가깝다.

### 3. DPO intuition: pairwise preference를 reference-relative margin으로 직접 민다
- DPO(Direct Preference Optimization)는 chosen / rejected pair를 중심으로 생각한다.
- 핵심 직관은 간단하다.
  - 현재 policy는 chosen 응답을 rejected보다 더 선호해야 한다.
  - 하지만 reference policy에서 너무 멀리 도망가면 style collapse, instability, knowledge drift가 생길 수 있다.
- 그래서 DPO는 보통 **현재 policy의 pairwise log-prob 차이** 를 **reference policy의 차이** 와 비교해, chosen 쪽 margin을 더 크게 만들도록 학습한다.
- high-level에서 보면 장점은 다음과 같다.
  - reward model을 별도로 학습하지 않아도 된다.
  - pairwise preference 데이터를 곧바로 objective로 사용할 수 있다.
  - RL rollout loop 없이도 SFT 이후 behavior shift를 만들 수 있다.
- 대신 pairwise data 품질에 민감하고, reference 대비 어느 정도만 벗어나게 할지(beta/regularization 직관)가 성격을 많이 바꾼다.

### 4. ORPO intuition: SFT anchor 위에 preference odds-ratio를 얹는다
- ORPO(Odds Ratio Preference Optimization)는 chosen 응답 likelihood를 유지·강화하는 SFT 성격과 preference ranking 성격을 한 번에 묶어 보려는 관점으로 이해할 수 있다.
- 실무 직관으로는 다음처럼 읽으면 된다.
  - chosen 응답은 그냥 잘 맞히게 한다.
  - 동시에 chosen이 rejected보다 더 높은 odds를 갖도록 추가 신호를 준다.
- 그래서 ORPO는 종종 **reference model을 따로 들고 가지 않는 단일-stage 정렬(monolithic alignment)** 감각으로 소개된다.
- high-level 장점/주의점은 다음과 같다.
  - 장점: SFT objective와 선호 objective를 한 프레임에서 묶기 쉽다.
  - 장점: reference policy 관리 부담을 줄일 수 있다.
  - 주의: chosen likelihood를 강하게 붙잡으면 pairwise separation보다 imitation 쪽으로 다시 끌릴 수 있다.
  - 주의: odds-ratio term이 실제로 얼마나 behavior shift를 만들었는지 별도 평가가 필요하다.

### 5. KTO intuition: strict pair가 아니라 desirability signal도 활용한다
- KTO(Kahneman-Tversky Optimization)는 prospect theory에서 이름을 가져온 계열로 소개되며, high-level에서는 **desirable / undesirable signal을 비대칭 효용처럼 다루는 preference optimization** 으로 이해하면 된다.
- 중요한 차이는 데이터 형식이다.
  - 반드시 chosen/rejected strict pair가 없어도 된다.
  - prompt-response 샘플에 대해 "이건 바람직함 / 바람직하지 않음" 같은 label을 붙일 수 있다.
- 그래서 KTO는 다음 상황에서 직관이 좋다.
  - pairwise annotation을 일관되게 만들기 어렵다.
  - safety/refusal/helpfulness처럼 undesirable 사례를 분리 표기하기 쉽다.
  - binary 또는 signed desirability signal이 더 자연스럽다.
- 대신 pairwise ordering이 직접 주는 비교 신호는 약해질 수 있고, label noise나 class imbalance가 objective 해석에 더 크게 들어올 수 있다.

### 6. DPO / ORPO / KTO high-level contrast
- 세 방법 모두 공통적으로 다음 목표를 가진다.
  - SFT 뒤 policy를 더 선호되는 응답 쪽으로 이동시킨다.
  - reward model + online PPO라는 무거운 RLHF loop를 일부 생략하거나 단순화한다.
  - post-training alignment를 loss 설계 문제로 가져온다.
- 하지만 차이는 분명하다.

#### 데이터 관점
- **DPO**: chosen/rejected pair 중심
- **ORPO**: chosen/rejected pair 중심
- **KTO**: desirable/undesirable label 중심, pair 필요성이 더 약함

#### regularization / anchor 관점
- **DPO**: reference policy 대비 과도한 이탈을 막는 감각이 핵심
- **ORPO**: chosen likelihood 자체가 anchor 역할을 하며 reference 의존이 약한 편
- **KTO**: desirability utility를 밀고 undesirable utility를 누르되, 구현에 따라 reference/KL 직관을 함께 볼 수 있음

#### operational trade-off 관점
- **DPO**: pairwise ranking 해석이 깔끔하지만 reference 관리와 pair 품질이 중요
- **ORPO**: one-stage 느낌이 좋아 pipeline을 단순화하기 쉽지만, 실제 preference separation과 imitation 신호의 균형을 따로 봐야 함
- **KTO**: label-only setup에 유연하지만, pairwise ordering만큼 강한 상대비교 신호는 약할 수 있음

### 7. alignment trade-offs: 선호를 밀면 다른 축이 같이 흔들릴 수 있다
- alignment objective는 흔히 helpfulness / harmlessness / honesty / style adherence 중 일부를 더 많이 밀어 준다.
- 이때 자주 생기는 trade-off는 다음과 같다.
  - 장황한 답변이 "더 친절해 보인다"는 이유로 과대선호됨
  - 안전성 선호가 강해져 필요한 답도 지나치게 거절함
  - 형식 충실도(format following)는 좋아졌지만 factuality는 그대로이거나 악화됨
  - 특정 annotator 취향(정중함, 길이, tone)이 전체 policy에 과도하게 박힘
  - domain-specific accuracy보다 generic assistant style이 더 강화됨
- 그래서 preference optimization의 성패는 win rate 하나로 끝나지 않는다. **어떤 축을 올렸고 무엇을 같이 잃었는지** 를 분리해서 봐야 한다.

### 8. evaluation concerns: offline 승리가 실제 품질을 보장하지는 않는다
- offline preference accuracy, judge-model win rate, chosen/rejected ranking accuracy는 유용하지만 충분하지 않다.
- 특히 다음 문제를 따로 봐야 한다.
  - **length bias**: 더 긴 답이 더 좋아 보이는가?
  - **position bias**: pair 순서/표현 방식에 따라 judge가 흔들리는가?
  - **style over substance**: 말투와 포맷이 사실성보다 과대평가되는가?
  - **safety regression**: harmful content 억제가 좋아졌는지, 아니면 over-refusal만 늘었는지?
  - **distribution shift**: 훈련 preference set과 다른 프롬프트 유형에서도 유지되는가?
- 결국 evaluation은 보통 묶어서 봐야 한다.
  - pairwise/judge win rate
  - task-specific accuracy / factuality
  - refusal/safety behavior
  - verbosity / latency / cost
  - human review spot checks

## Common Confusion
- chosen 응답을 "절대 정답"으로 보는 실수
- DPO / ORPO / KTO를 모두 같은 loss의 이름만 다른 버전으로 여기는 실수
- preference optimization이면 자동으로 RLHF보다 쉽고 항상 안정적이라고 생각하는 실수
- reference policy가 있으면 곧 reward model도 있어야 한다고 혼동하는 실수
- offline win rate가 오르면 실제 사용자 만족도도 자동으로 오른다고 믿는 실수
- safety 선호를 올리면 harmlessness만 좋아지고 helpfulness는 그대로라고 단정하는 실수
- pairwise 데이터가 없으면 preference optimization 자체가 불가능하다고 보는 실수

## 이 단위에서 무엇을 관찰할 것인가
- preference dataset에서 prompt / chosen / rejected / desirability label은 각각 어떤 의미를 갖고 어디서 노이즈가 들어오는가?
- DPO / ORPO / KTO는 각각 어떤 데이터 형식을 더 자연스럽게 요구하는가?
- reference policy나 anchor term은 policy drift를 줄이는 대신 어떤 보수성을 가져오는가?
- pairwise separation이 커졌을 때 format following, factuality, refusal balance는 같이 어떻게 변하는가?
- offline judge win rate가 length bias나 style bias에 얼마나 민감한가?
- 다음 RLHF 단위로 넘어갈 때, 어떤 한계 때문에 online rollouts와 reward modeling이 다시 필요해지는가?
