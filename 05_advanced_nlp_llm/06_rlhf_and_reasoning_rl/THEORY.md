# 06 RLHF and Reasoning RL 이론 노트

## 핵심 개념

### 1. reward model intuition: truth engine이 아니라 preference proxy다
- reward model은 보통 chosen / rejected pair나 ranked response, 혹은 rule/judge 기반 선호 신호를 보고 **어떤 응답이 더 바람직해 보이는지** 를 scalar로 근사하려는 모델이다.
- 중요한 점은 reward가 "정답 점수" 와 같지 않다는 것이다.
  - annotator 취향, rubric 설계, judge model 편향, 길이 선호가 reward 안에 섞일 수 있다.
  - factual correctness가 높아도 style이나 safety formatting이 annotation 기준과 다르면 reward가 낮을 수 있다.
  - 반대로 장황하고 그럴듯한 답이 실제 정확도보다 높은 reward를 받을 수도 있다.
- 그래서 reward model은 보통 이렇게 읽는 편이 안전하다.
  - **정답 판별기** 가 아니라 **현재 우리가 수집한 선호 기준의 압축 표현**
  - policy가 어느 방향으로 움직이고 있는지 보여 주는 신호이지, 진실의 최종 심판이 아님
- 이 직관이 있어야 reward hacking, verbosity inflation, over-refusal 같은 현상을 자연스럽게 해석할 수 있다.

### 2. RLHF high-level loop: offline pair에서 online behavior shaping으로 간다
고전적인 high-level RLHF loop는 대체로 다음 순서를 가진다.
1. SFT policy 혹은 이미 정렬된 초기 policy를 준비한다.
2. 프롬프트를 뽑아 현재 policy로 여러 응답 후보를 생성한다.
3. 사람/규칙/judge/기존 선호 데이터를 이용해 reward signal을 만든다.
4. 그 reward를 바탕으로 policy를 update한다. 구현은 PPO 계열, advantage-style update, GRPO류 등 다양할 수 있지만 핵심은 **높은 reward 방향으로 policy distribution을 다시 미는 것** 이다.
5. update 이후에는 factuality, safety, refusal, latency, verbosity, format-following regression을 다시 본다.

이 루프가 중요한 이유는 policy가 변할수록 **모델이 실제로 내는 응답 분포도 바뀌기 때문** 이다.
- offline preference dataset만 보면 이전 policy나 수집 시점의 데이터 분포에 묶이기 쉽다.
- online rollout을 하면 **현재 policy가 실제로 뱉는 이상한 답변·새로운 failure mode** 를 다시 볼 수 있다.
- 대신 online RLHF는 비용이 크고, reward model이 불완전하면 잘못된 방향으로 빠르게 최적화될 위험도 커진다.

즉 RLHF는 "더 강력한 최적화" 인 동시에 "더 위험한 최적화" 라고 보는 편이 맞다.

### 3. reasoning-oriented RL framing: 최종 답만이 아니라 추론 행동을 어떻게 밀 것인가
reasoning-oriented RL은 단순히 "정답이면 보상, 오답이면 벌점" 으로 끝나지 않는다. 관심사는 보통 다음에 있다.
- 문제를 바로 찍지 않고 단계적으로 풀려 하는가?
- self-correction이나 backtracking이 필요한 상황에서 다시 검토하는가?
- verifier / tool / search 결과를 보고 답을 수정하는가?
- reasoning trace가 길기만 한 것이 아니라 **검증 가능한 중간 상태** 를 남기는가?

이때 보상 신호는 대략 세 부류로 나눠 생각할 수 있다.
- **outcome-only reward**: 최종 정답 여부, 최종 과업 점수만 본다.
- **process-aware reward**: intermediate step의 일관성, verifier 통과, 규칙 준수 같은 과정 신호를 함께 본다.
- **hybrid reward**: 최종 답 + 과정 품질 + format / safety 제약을 함께 섞는다.

핵심은 reasoning RL이 "긴 chain-of-thought를 무조건 더 쓰게 하는 것" 이 아니라는 점이다.
- 길어진 reasoning은 reward를 속이기 쉬운 표면 신호가 될 수 있다.
- 정말 원하는 것은 보통 **문제를 더 잘 분해하고, 더 잘 검토하고, 더 잘 수정하는 행동** 이다.
- 그래서 reasoning RL에서는 과정 신호를 넣더라도 "길이" 대신 "검증 가능성" 과 "오류 수정 능력" 을 더 중요하게 봐야 한다.

### 4. verifier / judge interaction: 좁은 검사와 넓은 비교가 만나는 지점
reasoning RL이나 RLHF 실무에서는 verifier와 judge가 자주 같이 등장한다.

#### verifier의 high-level 역할
- verifier는 보통 더 좁고 구조화된 질문을 다룬다.
  - 수학 풀이의 중간 식이 맞는가?
  - 형식 제약을 지켰는가?
  - tool result와 final answer가 모순되지 않는가?
- 장점은 signal이 더 국소적이고 자동화하기 쉽다는 점이다.
- 단점은 "체크리스트를 통과하는 척" 하는 gaming에 취약하고, verifier가 검사하지 않는 오류는 그대로 남는다는 점이다.

#### judge의 high-level 역할
- judge는 보통 두 응답을 비교하거나 rubric 기반으로 더 넓은 품질 판단을 한다.
  - 어떤 답이 더 helpful한가?
  - 설명과 정답을 함께 보면 어느 쪽이 더 설득력 있는가?
  - 안전성/거절/스타일 관점에서 어느 쪽이 더 바람직한가?
- 장점은 넓은 품질을 한 번에 볼 수 있다는 점이다.
- 단점은 length bias, style bias, position bias에 흔들리기 쉽고, 정답성보다 표현력에 과대 반응할 수 있다는 점이다.

#### 둘의 상호작용을 어떻게 읽을 것인가
- verifier score가 높고 judge win rate도 높으면 강한 신호처럼 보이지만, 여전히 factual check가 필요하다.
- verifier는 통과했는데 judge는 낮게 본다면, **형식은 맞았지만 실제 답변 경험이 별로** 일 수 있다.
- judge는 높게 봤는데 verifier는 자주 실패한다면, **겉보기 품질은 좋지만 내부 일관성이나 계산 정확성이 약할 수 있다.**
- 결국 둘은 대체재라기보다 서로 다른 blind spot을 가진 신호원이다.

### 5. common confusion: RLHF와 reasoning RL에서 자주 헷갈리는 것
- reward model을 truth model로 보는 실수
- RLHF를 "사람 라벨 + PPO" 라는 고정된 공식으로만 생각하는 실수
- online RL을 쓰면 offline preference optimization보다 항상 낫다고 믿는 실수
- reasoning RL이 곧 "더 긴 chain-of-thought 생성" 이라고 생각하는 실수
- verifier가 통과하면 reasoning도 옳다고 단정하는 실수
- judge win rate 상승을 실제 사용자 만족도나 안전성 개선과 곧바로 동일시하는 실수
- process reward를 넣으면 reward hacking이 줄어든다고 자동으로 믿는 실수
- reasoning quality를 올리면 latency / cost / verbosity trade-off는 그대로일 것이라고 생각하는 실수

### 6. 이 단위에서 무엇을 관찰할 것인가
- reward model이 어떤 선호를 압축하고 있고, 어떤 축을 놓치고 있는가?
- online rollout 이후 새롭게 등장하는 failure mode는 무엇인가?
- reward가 높아질수록 답변 길이, refusal, style, format은 어떤 방향으로 흔들리는가?
- outcome reward와 process-aware reward는 서로 어떤 문제를 보완하고 어떤 왜곡을 추가하는가?
- verifier와 judge가 자주 불일치하는 slice는 어디이며, 그 불일치가 무엇을 말해 주는가?
- "reasoning이 좋아졌다" 는 말을 할 때 최종 정답률, verifier consistency, judge preference, latency를 어떻게 함께 읽어야 하는가?
- 다음 retrieval/eval 단위로 넘어갈 때, 내부 보상 최적화만으로는 해결되지 않는 grounding 문제는 무엇인가?
