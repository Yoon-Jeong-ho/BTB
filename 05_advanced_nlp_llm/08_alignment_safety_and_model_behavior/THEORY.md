# 08 Alignment, Safety, and Model Behavior 이론 노트

## 핵심 개념
이 단위의 핵심은 **alignment vs capability**를 분리하고, 안전한 모델 행동을 **refusal vs over-refusal**, **harmlessness / robustness**, **behavioral eval slice analysis**, **policy vs system-level safety**로 관찰하는 것이다. 실제 LLM을 학습하지 않아도, 작은 toy eval만으로 어떤 평균 점수가 어떤 실패를 숨길 수 있는지 볼 수 있다.

### 1. alignment vs capability: 할 수 있음과 해야 함은 다르다
- capability는 모델이 어떤 작업을 원리적으로 수행할 수 있는지에 가깝다.
  - 사실 질문에 답하기
  - 요약하기
  - 코드 작성하기
  - 다단계 추론하기
- alignment는 그 능력이 실제 사용자 상호작용에서 어떤 정책, 맥락, 시스템 제약 아래 어떤 행동으로 나타나는지를 다룬다.
- capable but poorly aligned 모델은 유용한 정보를 생성할 능력은 높지만 위험한 요청도 지나치게 잘 수행할 수 있다.
- aligned-looking but incapable 모델은 안전한 말투를 쓰지만 실제 문제 해결력은 낮을 수 있다.
- 따라서 capability benchmark와 deployment behavior benchmark는 서로 다른 질문이다.

### 2. refusal vs over-refusal: 거절률 하나로 안전성을 말할 수 없다
- **refusal**은 특정 요청에 응답하지 않거나 제한된 형태로만 응답하는 행동이다.
- harmful request에 대한 refusal은 보통 바람직한 harmlessness 행동이다.
- **over-refusal**은 허용 가능하거나 유익한 요청까지 지나치게 거절하는 현상이다.
- over-refusal은 harmlessness를 높인 것처럼 보이지만 실제로는 helpfulness, trust, task completion을 깎는다.
- 좋은 안전 행동은 무조건 거절이 아니라 다음 균형이다.
  - benign request: 답한다.
  - harmful request: refuse and redirect한다.
  - borderline request: 범위를 제한하고 safe alternative를 제시한다.

### 3. harmlessness와 robustness
- harmlessness는 위험한 내용을 직접 돕지 않는 것뿐 아니라 안전한 대안, 범위 제한, 불확실성 표현, 취약한 상황 보호까지 포함한다.
- robustness는 policy behavior가 입력 표현 변화에도 안정적인지를 본다.
  - paraphrase를 해도 같은 판단을 내리는가?
  - formatting noise나 다국어 표현이 들어가도 기준이 유지되는가?
  - jailbreak-style phrasing이나 role confusion이 들어와도 쉽게 무너지지 않는가?
- robustness가 낮으면 평균 harmlessness score가 좋아도 배포에서는 prompt phrasing에 따라 행동이 흔들릴 수 있다.

### 4. behavioral eval과 slice analysis
단일 정확도나 judge score는 alignment/safety를 충분히 설명하지 못한다. behavioral eval은 보통 slice analysis로 읽어야 한다.

- helpfulness slice: benign request에서 답해야 할 것을 답하는가?
- harmful refusal slice: clearly harmful request에서 안정적으로 refusal하는가?
- over-refusal slice: 허용 가능한 요청을 불필요하게 막지 않는가?
- borderline slice: 안전한 축소 응답이나 safe alternative를 제시하는가?
- robustness slice: paraphrase / noise / jailbreak-style variant에서도 behavior가 유지되는가?
- policy consistency slice: 같은 정책 경계가 다른 표현에서도 일관적인가?

주의할 점은 다음과 같다.
- 높은 refusal rate는 harmlessness 개선이 아니라 over-refusal 악화일 수 있다.
- judge win rate는 길고 정중한 답변에 편향될 수 있다.
- 안전해 보이는 phrasing이 실제로는 위험한 핵심 정보를 우회 제공할 수 있다.
- offline eval이 좋아도 실제 트래픽 분포에서 같은 행동을 보장하지는 않는다.

### 5. policy vs system-level safety
안전성은 모델 정책만으로 끝나지 않는다. **policy vs system-level safety**를 분리해야 한다.

#### model policy가 맡는 것
- unsafe content refusal
- safe alternative phrasing
- uncertainty handling
- conversational boundary 유지
- benign request의 과도한 거절 줄이기

#### system guardrail이 맡는 것
- tool permission gating
- auth / access control
- retrieval filtering
- moderation and audit logging
- rate limit과 abuse monitoring
- sandbox / execution isolation
- human review escalation

예를 들어 모델이 위험한 도구 호출을 거절하도록 학습되어 있어도, 실제 권한 없는 tool call이 시스템에서 실행되면 safety boundary는 실패한다. 반대로 system guardrail이 강해도 모델이 benign request를 계속 over-refuse하면 제품 경험은 무너진다.

## 실행 결과 예시와 해석
`scratch_lab.py`는 benign / harmful / borderline 요청을 직접 라벨링하고 `alignment_behavior_slices.svg`로 helpfulness, harmful refusal, safe alternative, robustness, over-refusal을 비교한다. `framework_lab.py`는 capability-only assistant와 aligned assistant를 deterministic behavior-eval simulation으로 비교한다.

핵심 질문은 숫자가 큰가가 아니라 다음이다.
- capability score가 높을 때 어떤 unsafe compliance 위험이 남는가?
- refusal이 harmlessness인지 over-refusal인지 어떤 slice에서 판단하는가?
- robustness probe가 낮은 경우 어떤 paraphrase나 jailbreak variant가 policy를 흔드는가?
- slice analysis가 단일 scalar보다 어떤 실패 모드를 더 잘 보여 주는가?
- model policy와 system guardrail 중 어느 층에서 막아야 할 실패인가?

## Common Confusion
- alignment를 capability 향상과 같은 말로 쓰는 실수
- refusal rate가 높으면 곧 안전성이 높다고 보는 실수
- harmlessness를 무조건 거절로 이해하는 실수
- robustness를 오직 jailbreak 저항성으로 좁히는 실수
- policy 문서만 잘 쓰면 system-level safety가 해결된다고 믿는 실수
- behavioral eval 평균 점수가 높으면 slice별 failure mode도 모두 해결됐다고 믿는 실수
