# 03 Domain Adaptive Pretraining 이론 노트

## 핵심 개념

### 1. continued pretraining intuition: 같은 objective를 유지한 채 분포를 다시 밀어 준다
- domain adaptive pretraining(DAPT)은 보통 **이미 일반 텍스트로 pretrain된 base LM** 에 대해, 같은 pretraining objective를 유지한 채 특정 도메인 corpus로 몇 step 더 학습하는 절차를 뜻한다.
- 핵심은 "처음부터 다시 학습"이 아니라 **기존 파라미터를 새로운 분포 쪽으로 조금 더 기울이는 것** 이다.
- 예를 들어 causal LM base model이라면 DAPT에서도 대개 causal LM objective를 유지한다. 바뀌는 것은 주로 다음이다.
  - 어떤 문서를 더 자주 보게 하는가
  - 얼마나 오래 적응시키는가
  - general replay를 섞는가
  - 어떤 validation으로 멈출 시점을 정하는가
- 그래서 DAPT는 구조를 바꾸는 단계라기보다 **분포 재가중(distribution reweighting)** 에 가깝다.

### 2. domain shift와 specialization: 무엇이 달라서 적응이 필요한가
- domain shift는 base model이 주로 봤던 일반 텍스트 분포와 새 도메인 데이터 분포 사이의 차이를 뜻한다.
- 이 차이는 단순 vocabulary 차이만이 아니라 다음처럼 여러 층위에서 나타난다.
  - 전문 용어 빈도와 co-occurrence 구조
  - 문서 형식(논문 초록, 판결문, SOAP note, 로그, 매뉴얼)
  - 문장 길이와 정보 밀도
  - 사실 표현 방식, 축약어, 표/코드/참조 구조
- DAPT가 잘 되면 모델은 이런 분포 차이에 더 빨리 적응해 in-domain perplexity나 downstream 성능이 좋아질 수 있다.
- 하지만 specialization은 공짜가 아니다. 한쪽 분포를 더 세게 밀수록 base model이 원래 잘하던 일반 표현 분포에서 조금 멀어질 수 있다.
- 따라서 DAPT의 질문은 보통 "도메인에서 얼마나 좋아졌는가?"와 동시에 "그 대가로 일반 능력을 얼마나 잃었는가?"를 함께 묻는다.

### 3. catastrophic forgetting trade-off: 더 잘 알게 되는 것과 덜 기억하게 되는 것 사이
- catastrophic forgetting은 새 분포에 맞추는 동안, 기존 분포에서의 유용한 파라미터 구성이 빠르게 무너지는 현상을 가리킨다.
- pure-domain continued pretraining은 적응 속도가 빠른 대신 forgetting 위험이 커지기 쉽다.
  - 장점: domain token distribution에 빠르게 맞는다.
  - 단점: general writing style, broad knowledge access, 잡다한 표현 복원력이 약해질 수 있다.
- 반대로 general replay나 mixed-domain sampling을 쓰면 stability는 좋아질 수 있지만 adaptation 속도는 느려질 수 있다.
- 직관적으로는 다음 두 힘이 충돌한다.
  - **plasticity**: 새 도메인에 빠르게 적응하려는 힘
  - **stability**: 기존 일반 분포 감각을 유지하려는 힘
- 실무에서는 이 균형을 다음 수단으로 조절한다.
  - domain-only vs mixed replay
  - learning rate / step budget 축소
  - high-quality small corpus oversampling
  - retention validation set 별도 추적
- 중요한 점은 forgetting이 단순히 "loss가 조금 올랐다"가 아니라 **모델의 사용 목적을 해칠 만큼 일반성이나 이전 능력이 줄었는가** 로 읽어야 한다는 것이다.

### 4. data selection concern: 아무 domain 데이터나 더 넣는다고 좋은 적응이 아니다
- DAPT용 corpus를 고를 때는 단순히 "그 도메인에서 수집했다"보다 더 많은 질문이 필요하다.
  - 정말 목표 사용 분포와 맞는가?
  - boilerplate / template / OCR noise / 정형 문서 반복이 과도하지 않은가?
  - evaluation 또는 downstream benchmark와 오염(contamination) 위험은 없는가?
  - 라이선스와 민감정보 처리 기준을 만족하는가?
  - recency가 중요한 도메인인지, 오래된 문서가 오히려 잘못된 신호를 주는지?
- 특히 전문 도메인에서는 데이터가 작고 비싸서, **무조건 많이 넣기보다 어떤 문서를 넣지 말아야 하는가** 가 더 중요할 때가 많다.
- 일반적으로는 다음 선택이 함께 따라온다.
  - pure-domain only
  - domain + general replay
  - broad-domain → narrow-domain curriculum
  - domain subset weighting by quality / recency / difficulty
- 결국 data selection은 DAPT의 절반 이상이다. 나쁜 corpus를 오래 학습하면 "도메인 적응"이 아니라 **노이즈 적응** 이 될 수 있다.

### 5. stopping concern: 더 오래 돌리는 것이 항상 더 좋은 것은 아니다
- DAPT에서는 training loss가 계속 떨어져도, 실제로는 이미 overspecialization이 시작됐을 수 있다.
- 그래서 stop 시점은 보통 하나의 loss만 보고 정하지 않는다.
  - in-domain validation loss / perplexity
  - general-domain retention loss
  - downstream probe task 성능
  - generation quality spot check
  - harmful regression guardrail
- 자주 보는 패턴은 다음과 같다.
  - 초반: in-domain metric이 빠르게 좋아짐
  - 중반: 개선 폭이 둔화됨
  - 후반: general retention이 더 나빠지는데 in-domain 이득은 작아짐
- 이때 "best step"은 단일 숫자가 아니라 **목표에 따라 다른 Pareto point** 가 된다.
- 즉 의료 QA 성능을 조금 더 올리는 것이 중요하다면 일반 도메인 성능 일부 하락을 감수할 수 있고, 범용 assistant를 유지해야 한다면 더 이른 stop이나 더 강한 replay가 필요하다.

### 6. DAPT는 instruction tuning과 다른 문제를 푼다
- DAPT는 주로 **지식 분포와 표현 분포를 적응** 시키는 단계다.
- instruction tuning/SFT는 **입력-출력 행동 형식과 응답 습관** 을 바꾸는 단계다.
- 그래서 DAPT가 잘 됐다고 곧바로 assistant답게 응답하는 것은 아니고, SFT가 잘 됐다고 도메인 지식이 자동으로 채워지는 것도 아니다.
- 실무에서는 둘을 분리해서 보는 편이 도움이 된다.
  - DAPT 질문: 이 모델이 이 도메인 텍스트를 더 잘 읽고 예측하는가?
  - SFT 질문: 그 지식을 사용자가 원하는 instruction format으로 잘 드러내는가?

## 수식 / 직관
- DAPT 이후 in-domain 개선을 거칠게 보면 다음처럼 쓸 수 있다.
  - `gain_in_domain ≈ loss_base(domain_val) - loss_adapted(domain_val)`
- 동시에 retention cost도 같이 본다.
  - `retention_cost ≈ loss_adapted(general_val) - loss_base(general_val)`
- 목표는 보통 `gain_in_domain`을 크게 만들면서 `retention_cost`를 허용 범위 안에 두는 것이다.
- 그래서 stop/mixture decision은 하나의 scalar 최적화라기보다 **이득과 손실의 동시 관리** 에 가깝다.

## 자주 헷갈리는 지점
- DAPT를 from-scratch pretraining의 축소판 정도로 이해하는 실수
  - 실제로는 기존 분포 감각을 가진 모델을 새 분포 쪽으로 미세하게 이동시키는 과정이다.
- DAPT와 instruction tuning을 같은 적응 단계로 보는 실수
  - 하나는 분포 적응이고, 다른 하나는 행동 형식 적응이다.
- domain corpus가 전문적일수록 무조건 많이 넣는 것이 좋다고 생각하는 실수
  - noisy template나 outdated 문서가 많으면 오히려 해가 될 수 있다.
- in-domain validation만 좋아지면 성공이라고 보는 실수
  - general retention과 downstream usability를 같이 봐야 한다.
- catastrophic forgetting을 "완전히 다 까먹는 극단적 현상"으로만 생각하는 실수
  - 실제로는 훨씬 작은 일반 성능 저하도 제품 목적상 치명적일 수 있다.
- stop 시점을 training loss 최저점 하나로 정하려는 실수
  - DAPT는 보통 multi-metric guardrail이 더 중요하다.

## 이 단위에서 무엇을 관찰할 것인가
- base corpus와 domain corpus 사이의 domain shift가 용어, 형식, 길이, 정보 밀도 측면에서 어떻게 드러나는가?
- pure-domain continued pretraining과 replay mixture는 adaptation speed와 forgetting profile을 어떻게 다르게 만드는가?
- in-domain validation이 계속 좋아지는 동안 general-domain retention은 언제부터 악화되기 시작하는가?
- data selection에서 고품질 소량 데이터 oversampling과 저품질 대량 데이터 투입은 어떤 다른 결과를 낳는가?
- stopping point를 loss, downstream probe, qualitative sample, guardrail 관점에서 함께 읽으면 어떤 결론이 달라지는가?
- 다음 단계 SFT 전에 DAPT가 정말 필요한 상황과, 오히려 건너뛰는 편이 나은 상황은 무엇인가?
