# 08 Alignment, Safety, and Model Behavior 이론 노트

## 핵심 개념

### 1. alignment vs capability: 무엇을 할 수 있는가와 어떻게 행동하는가는 다르다
- capability는 모델이 어떤 작업을 **원리적으로 수행할 수 있는지** 에 가깝다.
  - 사실 질문에 답하기
  - 요약하기
  - 코드 작성하기
  - 다단계 추론하기
- alignment는 그 능력이 실제 사용자 상호작용에서 **어떤 정책과 제약 아래 어떤 행동으로 나타나는지** 를 다룬다.
- 그래서 다음 두 상황은 모두 가능하다.
  - **capable but poorly aligned**: 유용한 정보를 생성할 능력은 높지만, 위험한 요청을 지나치게 잘 따라 준다.
  - **aligned-looking but incapable**: 말투는 안전하고 점잖지만, 실제로는 정확도와 문제 해결력이 낮다.
- 중요한 점은 alignment가 capability의 단순 하위 개념이 아니라는 것이다.
  - capability가 높아지면 위험한 요청을 수행할 잠재력도 함께 커질 수 있다.
  - 반대로 alignment 압력을 강하게 주면 benign task에서도 보수적으로 반응해 usefulness가 떨어질 수 있다.
- 따라서 이 단위에서는 "성능이 높은가" 와 "배포 가능한 행동을 보이는가" 를 의도적으로 분리해서 본다.

### 2. refusal / over-refusal / harmlessness / robustness: 안전 행동을 한 묶음으로 보되 섞어 말하지 않는다
- **refusal** 은 특정 요청에 응답하지 않거나 제한된 형태로만 응답하는 행동이다.
  - harmful request에 대한 refusal은 보통 안전 행동으로 본다.
  - 하지만 refusal 자체가 항상 좋은 것은 아니다.
- **over-refusal** 은 허용 가능하거나 유익한 요청까지 지나치게 거절하는 현상이다.
  - 예: 일반적인 보안 개념 설명, 합법적 연구 목적의 방어 조언, 정상적인 의료/법률 정보 요청을 모두 기계적으로 거절함
  - over-refusal은 harmlessness를 올리는 대신 usefulness와 trust를 깎는다.
- **harmlessness** 는 단순 거절보다 더 넓다.
  - 위험한 내용을 직접 돕지 않기
  - 안전한 대안 제시하기
  - 불확실한 경우 범위를 축소하거나 경고와 함께 응답하기
  - 사용자의 취약한 상황을 악용하지 않기
- **robustness** 는 정책 행동이 입력 표현 변화에도 안정적인지를 본다.
  - paraphrase를 해도 같은 안전 판단을 내리는가?
  - formatting noise나 multilingual prompt에서도 기준이 유지되는가?
  - jailbreak-style phrasing이나 role confusion이 들어와도 쉽게 무너지지 않는가?
- 즉 좋은 안전 행동은 "거절률이 높다" 가 아니라, **허용해야 할 것은 허용하고, 막아야 할 것은 막고, 표현이 흔들려도 판단 기준이 크게 무너지지 않는 상태** 에 더 가깝다.

### 3. behavioral eval intuition: accuracy 하나가 아니라 행동 slice를 본다
- alignment/safety 평가는 보통 단일 정확도 점수로는 부족하다.
- 더 자연스러운 질문은 다음에 가깝다.
  - benign request에서 충분히 helpful한가?
  - clearly harmful request에서는 안정적으로 refuse하는가?
  - borderline request에서는 지나친 거절 없이 안전한 축소 응답을 하는가?
  - prompt phrasing이 바뀌어도 같은 판단을 유지하는가?
- 그래서 behavioral eval은 보통 **slice-based evaluation** 으로 보는 편이 직관적이다.
  - helpfulness slice
  - harmful refusal slice
  - over-refusal slice
  - jailbreak / perturbation robustness slice
  - policy consistency slice
- 이때 주의할 점도 있다.
  - 높은 refusal rate는 harmlessness 개선이 아니라 over-refusal 악화일 수 있다.
  - judge model win rate는 길고 정중한 답변에 편향될 수 있다.
  - 안전한 phrasing이 좋아 보여도 실제로 위험한 핵심 정보를 우회 제공했을 수 있다.
  - offline eval이 좋아도 실제 배포 트래픽 분포에서 같은 행동을 보장하지는 않는다.
- 따라서 behavioral eval의 직관은 "좋아 보이는 평균 응답" 을 보는 것이 아니라, **어떤 유형의 입력에서 어떤 실패 모드가 얼마나 남는가** 를 보는 데 있다.

### 4. policy vs system-level safety boundaries: 모델만으로 해결할 수 없는 안전이 있다
- 안전성은 모델 내부 행동 정책만으로 끝나지 않는다.
- 거칠게 나누면 다음 두 층을 구분할 수 있다.

#### 모델/정책 레벨
- unsafe request에 대한 refusal phrasing
- uncertainty 표현
- harmless alternative 제시
- format adherence, style constraint, conversational boundary 유지

#### 시스템/제품 레벨
- tool permission gating
- auth / access control
- retrieval source filtering
- moderation endpoint, rate limit, audit logging
- user segmentation, human review escalation
- sandbox / execution isolation

- 왜 이 구분이 중요한가?
  - 어떤 안전 요구사항은 모델이 아무리 잘 거절해도 시스템이 뚫리면 무력화된다.
  - 반대로 시스템 guardrail이 있어도 모델이 benign request를 과도하게 거절하면 사용자 경험이 무너진다.
- 예를 들어,
  - PII가 포함된 도구 호출을 막는 일은 보통 시스템 레벨 권한 관리가 더 중요하다.
  - 위험한 표현을 직접 생성하지 않도록 거절/완화하는 일은 모델 정책도 중요하다.
- 따라서 alignment 논의는 "모델이 다 알아서 안전해야 한다" 가 아니라, **모델 정책과 시스템 guardrail이 각자 무엇을 맡는지 분리해서 설계하는 문제** 로 이해하는 편이 맞다.

### 5. capability improvement와 safety improvement는 같은 축으로 합산되지 않는다
- capability를 키우는 변화는 때로 더 정교한 위험 도움도 가능하게 만든다.
- safety를 강화하는 변화는 때로 benign task에서 과도한 거절을 낳는다.
- 그래서 post-training 개선을 볼 때는 최소한 다음 질문을 같이 던져야 한다.
  - 유용성이 올랐는가?
  - 위험한 도움은 줄었는가?
  - benign task 오탐 거절은 늘지 않았는가?
  - phrasing 변화에 대한 robustness는 유지되는가?
- 이 균형을 무시하면 "안전해졌다" 또는 "똑똑해졌다" 같은 단일 문장이 실제 행동 변화를 지나치게 단순화한다.

## Common Confusion
- alignment를 capability 향상과 같은 말로 쓰는 실수
  - 실제로는 능력 자체와 배포 행동 계약은 다른 축이다.
- refusal rate가 높으면 곧 안전성이 높다고 보는 실수
  - benign task까지 막으면 over-refusal일 수 있다.
- harmlessness를 "무조건 거절" 로 이해하는 실수
  - 안전한 대안, 범위 제한, 경고 포함 응답도 harmlessness의 일부다.
- robustness를 오직 jailbreak 저항성만으로 좁게 보는 실수
  - paraphrase, formatting noise, 다국어 표현 변화도 함께 봐야 한다.
- policy 문서만 잘 쓰면 시스템 안전 문제가 해결된다고 믿는 실수
  - 권한 관리, tool gating, logging 같은 system guardrail은 별도 설계가 필요하다.
- judge model 점수가 오르면 실제 제품 안전도도 자동으로 좋아졌다고 생각하는 실수
  - style bias, length bias, distribution shift 때문에 별도 slice 확인이 필요하다.

## 이 단위에서 무엇을 관찰할 것인가
- 같은 모델이 capability benchmark에서는 강하지만 behavioral eval에서는 왜 불안정할 수 있는가?
- harmful / benign / borderline 요청을 나눴을 때 refusal과 over-refusal은 어디서 갈리는가?
- harmless alternative를 제시하는 응답과 사실상 우회 도움을 주는 응답은 어떻게 구분할 것인가?
- paraphrase, role-play, formatting noise가 들어갔을 때 정책 판단은 얼마나 흔들리는가?
- 높은 judge score나 win rate 뒤에 숨어 있는 over-refusal, verbosity, style bias는 어떻게 드러낼 것인가?
- 모델 레벨 정책과 시스템 레벨 guardrail 중 무엇을 어느 층에 맡겨야 전체 안전성이 더 좋아지는가?
