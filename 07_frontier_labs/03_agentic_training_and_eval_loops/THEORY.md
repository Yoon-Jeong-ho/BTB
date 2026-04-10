# 03 Agentic Training and Eval Loops 이론 노트

## 핵심 개념

### 1. agentic loop는 train/eval workflow를 감싸는 운영 계층이다
- agentic training/eval loop를 흔히 "모델이 알아서 실험을 계속 돌리는 것"처럼 상상하지만, 실제로 중요한 것은 자동 실행보다 **의사결정 경계와 증거 수집 경로**다.
- training/eval workflow에는 원래도 여러 단계가 있다.
  - 무엇을 시도할지 정하는 planning
  - 실제 run을 수행하는 execution
  - 결과를 protocol에 맞게 확인하는 verification
  - 다음 행동을 고르는 critique / triage
- agentic loop는 이 네 단계를 한 번 더 문서화하고 반복 가능하게 만드는 운영 계층이다.
- 따라서 좋은 loop의 질문은 "얼마나 많이 돌렸는가"보다 **"각 반복이 같은 실험 계약 아래 해석 가능한가"** 에 가깝다.
- 이 관점이 없으면 agent는 job launcher일 뿐이고, 결과가 늘어날수록 오히려 해석 가능성은 떨어질 수 있다.

### 2. planner / executor / verifier / critic를 분리해야 self-approval를 줄일 수 있다
- agentic workflow에서 가장 위험한 구조는 한 역할이 계획도 세우고 실행도 하고 자기 결과를 스스로 승인하는 경우다.
- 그래서 최소한 논리적으로는 다음 역할을 분리해 생각해야 한다.

#### planner
- 이번 반복에서 무엇을 바꿀지, 무엇은 고정할지 정한다.
- acceptance gate, retry budget, stop condition을 먼저 명시한다.
- planner의 핵심 산출물은 코드가 아니라 **change set과 비교 기준이 적힌 실험 카드**다.

#### executor
- planner가 허용한 범위 안에서 실제 학습/평가 run을 수행한다.
- train log, eval metric, seed, config hash, hardware/runtime 메모를 남긴다.
- executor는 성능을 해석하지 않는다. 역할은 **정해진 계약을 충실히 실행하고 기록하는 것**이다.

#### verifier
- 현재 run이 정말 비교 가능한 run인지 확인한다.
- 같은 dataset/split/protocol을 썼는지, artifact가 빠지지 않았는지, baseline과 같은 경기장에서 비교 가능한지 검사한다.
- verifier의 목적은 칭찬이 아니라 **과장 방지**다. metric 개선이 있어도 protocol mismatch면 통과시키지 않는다.

#### critic
- verifier를 통과한 evidence를 읽고 다음 행동을 고른다.
- retry, rollback, ablation 분리, scope 축소, human escalation 중 무엇이 맞는지 판단한다.
- critic는 "무엇이 더 좋아 보이는가"보다 **무엇을 아직 주장할 수 없는가**를 먼저 적어야 한다.

- 이 분리가 중요한 이유는 두 가지다.
  - 첫째, planner가 세운 가설을 verifier가 그대로 승인하지 않게 하여 self-justification을 줄인다.
  - 둘째, critic가 실패를 단순 bug/성공 이분법으로 보지 않고 다음 iteration 설계 근거로 쓰게 만든다.

### 3. 좋은 iteration loop는 변화량과 종료 조건이 작고 명확하다
- frontier 실험에서 loop가 망가지는 가장 흔한 이유는 한 번의 iteration이 너무 많은 변수를 동시에 바꾸기 때문이다.
- 예를 들어 learning rate, batch size, negative sampling, eval prompt template를 한 번에 바꾸면, 결과가 좋아져도 왜 좋아졌는지 알 수 없다.
- 따라서 iteration loop는 보통 다음 구조를 가진다.
  1. contract freeze: 이번 run에서 고정할 benchmark / split / metric / budget 명시
  2. proposal: planner가 바꿀 변수와 예상 evidence를 한두 개로 제한
  3. execution: executor가 run 수행, 로그와 artifact 수집
  4. verification: artifact completeness와 protocol match 확인
  5. critique: 결과 해석, 다음 action 선택
  6. termination or escalation: 반복 종료 / 사람 검토 / benchmark 재설계 여부 결정
- 여기서 termination rule이 빠지면 retry storm가 생긴다.
- retry storm는 "조금만 더 돌리면 좋아질 것 같다"는 감정이 자동화된 상태다. 예산을 태우고도 왜 좋아졌는지 모르는 결과만 남기기 쉽다.
- 그래서 loop에는 최소한 다음 문장이 필요하다.
  - 몇 번까지 retry할 것인가?
  - 어떤 warning이 보이면 바로 human escalation할 것인가?
  - variance band보다 작은 개선은 improvement로 인정할 것인가?
  - benchmark drift나 artifact 누락이 나오면 재시도 대신 중단할 것인가?

### 4. evidence collection은 metric 수집보다 넓은 계약이다
- 많은 팀이 evidence를 metric json 하나로 축소해서 생각한다.
- 하지만 agentic workflow에서 evidence bundle은 최소한 다음을 포함해야 한다.
  - experiment contract ID와 change set
  - dataset / split / version / prompt template / preprocessing 메모
  - config hash, seed, dependency/runtime/hardware 정보
  - train/eval log와 핵심 metric
  - 실패 로그, timeout, OOM, skipped batch 같은 운영 신호
  - verifier checklist와 통과/보류 사유
  - critic verdict와 다음 행동 제안
- 이렇게 넓게 잡아야 하는 이유는, agentic loop의 주요 실패가 순수 metric 문제가 아닌 경우가 많기 때문이다.
- 예를 들어 metric이 올라도 benchmark contamination이 의심되면 improvement claim은 약해진다.
- 반대로 metric이 그대로여도 verifier가 artifact quality를 높이고 critic가 실패 원인을 잘 분리했다면, 그 iteration은 충분히 가치 있다.
- 즉 evidence collection의 목표는 "좋은 숫자 수집"이 아니라 **다음 반복이 같은 사실을 다시 읽을 수 있는 상태**를 만드는 것이다.

### 5. common failure modes: agentic loop는 속도와 함께 왜곡도 증폭한다
- agentic loop를 붙이면 실험이 빨라지는 대신, 잘못된 의사결정도 더 빨리 반복된다.
- 자주 보는 failure mode는 다음과 같다.

#### planner drift
- planner가 원래 capstone scope를 벗어나 점점 더 큰 변화를 제안한다.
- symptom: iteration이 지날수록 실험 질문보다 TODO 목록이 커진다.
- observation point: 현재 change set이 원래 acceptance gate와 같은 문제를 풀고 있는가?

#### executor overreach
- executor가 편의상 preprocessing, data slice, checkpoint rule까지 바꿔 버린다.
- symptom: run은 완료됐지만 baseline comparability가 사라진다.
- observation point: 이번 run에서 바뀐 변수 목록이 planner brief와 정확히 일치하는가?

#### verifier shallowness
- verifier가 metric만 보고 artifact 누락이나 protocol mismatch를 놓친다.
- symptom: 점수는 좋아 보이지만 재현 가능한 근거가 없다.
- observation point: seed, split, config, artifact path, baseline match가 checklist에 실제로 채워졌는가?

#### critic hallucination
- critic가 소수의 결과만 보고 과한 가설을 세운다.
- symptom: "데이터 부족이 원인 같다"처럼 plausible하지만 검증되지 않은 설명이 누적된다.
- observation point: critic verdict가 실제 evidence field를 인용하는가, 아니면 추측 서술에 머무는가?

#### retry storm
- 작은 개선 신호에 취해 계속 비슷한 실험을 반복한다.
- symptom: iteration 수는 늘지만 새로운 정보량은 줄어든다.
- observation point: 최근 N회 run 중 새로 배운 사실이 무엇인지 한 줄로 설명 가능한가?

#### benchmark contamination
- loop가 benchmark 특이 규칙을 우연히 과적합한다.
- symptom: offline metric은 오르는데 qualitative failure나 holdout slice는 악화된다.
- observation point: verifier/critic가 benchmark drift, label leakage, evaluator dependency를 따로 점검하는가?

### 6. 이 단위에서 봐야 할 관찰 포인트는 '좋은 run'보다 '정직한 run'이다
- agentic workflow를 잘 설계했는지 보려면 단순 최고 metric보다 다음 관찰 포인트를 본다.
  - 각 iteration이 같은 contract 아래 비교 가능한가?
  - 역할 분리가 실제로 log와 verdict에 남아 있는가?
  - artifact 누락, protocol mismatch, benchmark drift가 조기에 잡히는가?
  - retry와 escalation 규칙이 문장으로 고정되어 있는가?
  - critic가 다음 action을 evidence 기반으로 추천하는가?
  - 사람 검토가 필요한 순간을 loop가 숨기지 않는가?
- 결국 좋은 agentic loop는 "계속 돌았다"가 아니라, **무엇을 자동화했고 무엇을 인간 판단으로 남겼는지**, **왜 지금 멈췄는지**, **무슨 증거가 남았는지**를 설명할 수 있어야 한다.

## 자주 헷갈리는 지점
- agentic loop를 단순 job scheduler와 같은 개념으로 보는 실수
- planner와 critic를 분리하지 않아 가설과 판정이 한 목소리가 되는 실수
- verifier를 metric checker로만 축소해 artifact / protocol 검증을 놓치는 실수
- retry 횟수가 많으면 탐색이 잘 되고 있다고 착각하는 실수
- benchmark 자체가 흔들리는데도 loop만 더 촘촘히 만들면 해결될 것이라고 믿는 실수
- 사람 개입이 있으면 agentic하지 않다고 오해하는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 현재 loop contract가 문제 정의, benchmark, budget을 충분히 고정하고 있는가?
- planner가 한 iteration에서 바꾸는 변수 수가 해석 가능한 수준으로 제한되어 있는가?
- executor run이 끝난 뒤 다음 사람이 바로 읽을 수 있을 정도로 artifact bundle이 남는가?
- verifier checklist가 실제로 과장된 improvement claim을 막는가?
- critic verdict가 evidence 인용과 함께 retry / stop / escalate를 구분하는가?
- 다음 단위의 benchmark/dataset construction과 연결될 만큼 benchmark drift 관찰 포인트가 선명한가?
