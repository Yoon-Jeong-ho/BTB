# 03 DeepSpeed ZeRO 이론 노트

## 핵심 개념

### 1. ZeRO partitioning intuition: "모든 rank가 모든 상태를 다 들고 있을 필요가 있는가?"
- 기본 DDP에서는 각 rank가 모델 파라미터와 gradient, 그리고 optimizer state를 사실상 전부 들고 있는 경우가 많다.
- 이 구조는 구현과 사고가 단순한 대신, 모델이 커질수록 **중복 복사(replication)** 비용이 너무 커진다.
- ZeRO(Zero Redundancy Optimizer)의 핵심 직관은 "학습에 필요한 상태가 모두 매 step 동시에 모든 rank에 상주할 필요는 없다"는 데 있다.
- 즉, 필요한 순간에만 모으고 평소에는 나눠 들면 **정확한 학습 의미를 유지하면서 중복 메모리만 줄일 수 있다**.
- 그래서 ZeRO는 새로운 모델 구조라기보다, distributed optimizer/runtime이 **상태 저장 방식을 재배치하는 메모리 전략**에 가깝다.

### 2. 메모리 세 덩어리: optimizer state / gradient / parameter
분산 학습 메모리를 볼 때 최소한 아래 세 항목을 따로 봐야 한다.

#### optimizer state
- Adam 계열이면 보통 parameter 자체 외에 1차/2차 모멘트가 추가로 필요하다.
- 그래서 large model에서는 종종 **optimizer state가 가장 큰 메모리 덩어리**가 된다.
- Stage 1이 먼저 여기를 partition하는 이유도, 구현 난이도 대비 절감 효과가 크기 때문이다.

#### gradient
- backward 이후 optimizer step 직전까지 유지되는 gradient도 큰 메모리를 차지한다.
- DDP에서는 all-reduce 전후로 gradient를 rank마다 사실상 모두 다루는 구조를 떠올리기 쉽다.
- ZeRO Stage 2는 이 gradient까지 나눠 들며, 통신 패턴도 더 적극적으로 분산 최적화 쪽으로 이동한다.

#### parameter
- parameter는 forward/backward 내내 참조되는 핵심 상태다.
- 그래서 이것까지 나누기 시작하면 절감 폭은 커지지만, runtime orchestration은 훨씬 더 복잡해진다.
- ZeRO Stage 3는 parameter까지 shard하므로 가장 공격적인 메모리 절감이 가능하지만, all-gather/re-materialization 타이밍을 더 세심하게 관리해야 한다.

### 3. Stage 1 / 2 / 3를 한 줄씩 잡기
- **Stage 1**: optimizer state partitioning
  - 파라미터와 gradient는 복제 상태를 유지하면서, optimizer state만 rank마다 나눠 든다.
  - 가장 보수적이지만 체감 메모리 이득이 큰 첫 단계다.
- **Stage 2**: optimizer state + gradient partitioning
  - backward 이후 gradient 저장 비용까지 줄인다.
  - 통신과 runtime 흐름이 더 복잡해지지만 대형 batch/대형 모델에서 메모리 완화가 더 크다.
- **Stage 3**: optimizer state + gradient + parameter partitioning
  - 가장 큰 절감 폭을 주는 대신, forward/backward 시점마다 필요한 parameter shard를 모으는 orchestration 부담이 커진다.
  - 그래서 stage 숫자는 "고급 버전"이라기보다 **더 깊은 partitioning과 더 큰 coordination 비용**을 의미한다.

### 4. 왜 DeepSpeed가 큰 모델에서 중요한가
- large model training에서는 batch보다 먼저 **optimizer state 메모리**가 터지는 경우가 흔하다.
- gradient accumulation만 늘리거나 micro-batch를 줄여도, optimizer state 복제 비용 자체는 사라지지 않는다.
- DeepSpeed는 ZeRO를 통해 이 중복 메모리를 체계적으로 줄여 주고, mixed precision, accumulation, offload, launcher/runtime 구성까지 함께 엮어 준다.
- 즉 "큰 모델을 겨우 올리는 트릭"이 아니라, **한정된 GPU 메모리 안에서 학습 계약을 재구성하는 운영 프레임워크**로 중요하다.
- later LLM 학습에서 DeepSpeed가 반복적으로 등장하는 이유도, 모델이 커질수록 연산만큼이나 **상태를 어디에 둘 것인가**가 핵심 문제가 되기 때문이다.

### 5. 단순한 setup과 비교한 trade-off

#### DDP + 작은 batch / grad accumulation과 비교
- 장점
  - 개념과 디버깅이 단순하다.
  - 통신 패턴과 failure mode가 상대적으로 예측 가능하다.
- 한계
  - optimizer state와 parameter replication이 그대로 남는다.
  - micro-batch를 아무리 줄여도 "상태 복제" 병목은 해결되지 않는다.

#### ZeRO의 장점
- per-rank memory 절감이 크다.
- 동일 GPU 수로 더 큰 모델 또는 더 큰 effective batch를 다루기 쉬워진다.
- large-scale training에서 운영 가능 범위를 넓힌다.

#### ZeRO의 비용
- config와 runtime이 복잡해진다.
- 통신 오버헤드가 눈에 띄기 시작한다.
- checkpoint, resume, optimizer state handling, debugging이 단순 DDP보다 까다로워진다.
- stage가 올라갈수록 memory saving은 커지지만, 어떤 collective가 언제 일어나는지 이해해야 할 일이 많아진다.

### 6. communication vs memory: 공짜 절약은 아니다
- ZeRO가 줄이는 것은 주로 **중복 저장 메모리**다.
- 대신 분산된 상태를 필요한 순간에 모으고 흩어야 하므로 all-gather, reduce-scatter 같은 collective가 더 중요해진다.
- 따라서 stage를 올리면 보통 다음 질문을 함께 봐야 한다.
  - per-rank memory는 얼마나 줄었는가?
  - step time은 얼마나 늘었는가?
  - interconnect bandwidth/NVLink/PCIe 차이가 병목으로 올라오는가?
  - 계산보다 통신 대기가 더 커지는가?
- 결국 좋은 설정은 "가장 높은 ZeRO stage"가 아니라, **현재 하드웨어/모델/배치 조건에서 메모리와 throughput의 균형이 맞는 지점**이다.

### 7. common confusion
- ZeRO를 단순히 "optimizer만 빠르게 만드는 라이브러리"로 이해하는 실수
  - 본질은 redundant state를 줄이는 메모리/runtime 전략이다.
- Stage 3가 항상 Stage 1/2보다 무조건 낫다고 생각하는 실수
  - 메모리는 더 아낄 수 있지만 통신/복잡도 비용이 커져 항상 최선은 아니다.
- gradient accumulation을 늘리면 ZeRO가 필요 없어진다고 생각하는 실수
  - accumulation은 activation/micro-batch 쪽 압박을 줄일 수 있지만 optimizer state replication은 그대로 남는다.
- parameter, gradient, optimizer state를 하나의 메모리 덩어리처럼 뭉뚱그리는 실수
  - 생성 시점, 유지 시간, partition 난이도가 서로 다르다.
- DeepSpeed config를 "값만 맞추면 되는 JSON"으로 생각하는 실수
  - 각 필드는 memory, communication, numerical stability, throughput에 직접 연결된 운영 계약이다.

### 8. 이 단위에서 무엇을 관찰할 것인가
- DDP 기준 memory breakdown에서 무엇이 가장 큰 덩어리로 보이는가?
- Stage 1/2/3로 갈수록 optimizer state, gradient, parameter 중 무엇이 언제 shard되는가?
- memory saving이 커질수록 어떤 collective communication이 더 자주 보이는가?
- same effective batch를 유지할 때 ZeRO와 grad accumulation-only setup은 어떤 다른 병목을 남기는가?
- DeepSpeed config에서 `zero_stage`, micro-batch, accumulation, offload를 함께 볼 때 어떤 운영 의사결정이 드러나는가?
- 다음 FSDP/offload 단위로 넘어갈 때, "state partitioning"과 "parameter lifecycle orchestration"을 어떻게 구분해서 볼 것인가?
