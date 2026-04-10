# 06 Pipeline Parallelism 이론 노트

## 핵심 개념

### 1. pipeline stage intuition: 레이어 스택을 시간축 위에서 나눠 본다
- data parallel은 같은 모델 복제본을 여러 rank가 들고 서로 다른 데이터를 처리한다.
- tensor parallel은 한 레이어 내부의 큰 행렬 연산 자체를 여러 장치가 나눠 계산한다.
- pipeline parallel은 이 둘과 달리, **레이어 순서 자체를 stage라는 연속 구간으로 잘라 여러 장치에 배치**한다.
- 예를 들어 24개 transformer block이 있다면 `0~5`, `6~11`, `12~17`, `18~23`처럼 네 stage로 나누는 식이다.
- 이때 각 stage는 자기 구간의 forward/backward만 책임지고, stage 경계에서는 activation을 다음 stage로 넘긴다.
- 그래서 pipeline parallel의 핵심 질문은 "GPU 수를 늘릴 수 있는가"보다, **순차 모델을 어디서 잘라야 stage별 compute와 memory가 균형을 이루는가**에 가깝다.

### 2. 왜 pipeline parallel이 필요한가
- 모델이 커지면 한 장치에 모든 레이어 파라미터와 activation을 동시에 올리기 어렵다.
- FSDP나 ZeRO는 상태 복제 비용을 줄이는 데 강하지만, 레이어 실행 구간 자체를 여러 장치로 나눠 들고 싶을 때는 pipeline parallel이 더 직접적인 도구가 된다.
- 특히 매우 깊은 transformer에서는 "레이어 일부만 각 장치에 두고 순서대로 흐르게 한다"는 발상이 메모리 적재 한계를 넘는 데 중요하다.
- 하지만 pipeline parallel은 공짜가 아니다.
  - stage 사이 activation transfer가 필요하다.
  - stage가 비어 있는 warmup/cooldown 구간이 생긴다.
  - stage 시간 불균형이 있으면 빠른 장치가 기다리게 된다.
- 따라서 pipeline parallel은 **메모리를 나눠 들게 해 주는 대신, 시간축 스케줄링 문제를 새로 만든다**고 이해하는 편이 정확하다.

### 3. bubble / throughput trade-off: pipeline은 채워지기 전과 비워질 때 빈 구간이 있다
- pipeline을 처음 시작할 때는 stage 0만 일하고 뒤 stage들은 아직 input을 기다린다. 이것이 warmup이다.
- 마지막 microbatch가 앞 stage를 떠난 뒤에는 뒤쪽 stage들만 남아 backward/마무리를 하게 되는데, 이것이 cooldown이다.
- 이 warmup + cooldown에서 일부 stage는 놀게 되며, 이런 idle 구간을 흔히 **pipeline bubble**이라 부른다.
- 중요한 직관은 다음과 같다.
  - microbatch가 적으면 bubble 비율이 커진다.
  - microbatch가 늘면 steady-state 구간이 길어져 bubble 비율은 줄어든다.
  - 하지만 microbatch를 무한히 늘리면 scheduling overhead, activation bookkeeping, launch overhead가 커진다.
- 그래서 pipeline parallel의 목표는 단순히 "많이 쪼개기"가 아니라, **bubble을 줄일 만큼 microbatch를 확보하면서도 메모리/오버헤드가 감당되는 지점을 찾는 것**이다.
- 또한 pipeline parallel은 보통 **단일 샘플 latency를 줄이는 기술**이 아니라, pipeline이 채워진 뒤의 throughput을 개선하는 기술에 더 가깝다.

### 4. microbatch scheduling basics: GPipe와 1F1B를 구분해서 보기

#### GPipe 스타일 직관
- 여러 microbatch를 앞에서부터 끝까지 forward로 먼저 밀어 넣고, 이후 backward를 한꺼번에 수행하는 감각이다.
- 이해가 비교적 단순하고 구현 설명이 쉽다.
- 하지만 forward가 끝날 때까지 backward를 못 하므로, activation을 오래 많이 들고 있어야 해 메모리 부담이 커질 수 있다.

#### 1F1B(one-forward-one-backward) 스타일 직관
- warmup 이후에는 가능한 한 forward와 backward를 교차시켜, 각 stage가 microbatch 하나의 forward를 처리한 뒤 다른 microbatch의 backward를 이어서 처리하는 식이다.
- activation을 오래 쌓아 두지 않도록 도와 메모리 부담을 줄이는 데 유리하다.
- 대신 스케줄 reasoning이 더 복잡하고, 어느 시점에 어떤 microbatch가 어느 stage에서 무엇을 하는지 시간축 추적이 중요해진다.

#### scheduling에서 꼭 붙여서 볼 것
- microbatch 수
- stage 수
- warmup / steady state / cooldown 길이
- activation 보관량
- optimizer step이 실제로 언제 가능한지
- 즉 pipeline scheduling은 "실행 순서" 문제가 아니라 **메모리와 throughput을 동시에 결정하는 runtime policy**다.

### 5. activation transfer and partitioning concerns: stage 경계는 공짜 선이 아니다
- pipeline stage 사이에서는 hidden state나 activation tensor를 다음 장치로 넘겨야 한다.
- 따라서 stage boundary를 정할 때는 단순히 레이어 개수만 세지 않고 다음을 함께 봐야 한다.
  - 경계 activation의 shape와 dtype
  - send/recv 빈도와 payload 크기
  - skip connection이나 residual path가 boundary를 넘을 때 필요한 텐서 흐름
  - attention cache나 auxiliary output처럼 경계를 복잡하게 만드는 부가 상태
- partitioning도 단순 균등 분할이 정답이 아니다.
  - 레이어 수가 같아도 compute time은 다를 수 있다.
  - embedding, final projection, loss head처럼 특정 stage만 유난히 무거울 수 있다.
  - activation 크기가 비슷해 보여도 communication bandwidth 병목은 stage 위치에 따라 다르게 드러날 수 있다.
- 그래서 좋은 partition은 "레이어 수를 균등하게 자른다"보다, **stage별 compute / memory / communication을 함께 맞춘다**는 관점에서 봐야 한다.

### 6. pipeline parallel이 다른 병렬화 축과 어떻게 다른가
- data parallel: 모델은 복제되고 데이터가 나뉜다.
- tensor parallel: 한 레이어 내부 tensor 연산이 쪼개진다.
- ZeRO/FSDP: parameter/gradient/optimizer state 저장 위치를 나눠 든다.
- pipeline parallel: 레이어 실행 순서를 stage 구간으로 분할한다.
- large-model training에서는 이 축들이 서로 대체재가 아니라 보완재로 등장한다.
- 예를 들어 pipeline parallel만으로는 stage 내부 큰 matmul 메모리/통신 문제가 남을 수 있고, data parallel만으로는 한 장치에 레이어 전체를 못 올리는 문제가 풀리지 않을 수 있다.
- 따라서 이 단위는 pipeline을 "모든 문제의 해법"으로 보지 않고, **execution-path partitioning이라는 하나의 축**으로 정확히 자리 잡게 만드는 것이 중요하다.

### 7. common confusion
- pipeline parallel을 "레이어를 아무렇게나 여러 GPU에 나누면 된다"고 생각하는 실수
  - 실제로는 stage balance, activation transfer, schedule policy가 함께 맞아야 한다.
- microbatch를 늘리면 무조건 좋아진다고 생각하는 실수
  - bubble은 줄 수 있지만 activation bookkeeping, launch overhead, optimizer-step cadence 해석이 더 복잡해질 수 있다.
- throughput 개선과 latency 개선을 같은 말로 보는 실수
  - pipeline은 보통 steady-state throughput을 끌어올리는 방향이지, single input latency를 자동으로 줄여 주지 않는다.
- stage 수를 늘리면 항상 메모리도 throughput도 동시에 좋아진다고 생각하는 실수
  - 지나친 분할은 communication과 bubble을 늘리고 imbalance를 악화시킬 수 있다.
- pipeline parallel을 tensor parallel과 혼동하는 실수
  - 전자는 레이어 구간 분할, 후자는 레이어 내부 연산 분할이다.
- activation transfer를 단순 포인터 이동처럼 생각하는 실수
  - 실제로는 장치 간 통신이며, dtype/shape/ordering mismatch가 바로 runtime 문제로 드러난다.

### 8. 이 단위에서 무엇을 관찰할 것인가
- 어떤 모델 구간을 stage boundary로 잡았을 때 stage별 compute time과 memory가 얼마나 다르게 보이는가?
- microbatch 수를 바꾸면 warmup/cooldown 길이와 bubble fraction이 어떻게 변하는가?
- GPipe식과 1F1B식 스케줄에서 activation 보관량과 idle time이 어떤 차이를 보이는가?
- 어느 stage가 가장 자주 기다리며, 그 이유가 compute imbalance인지 communication boundary인지 구분할 수 있는가?
- activation send/recv가 어떤 텐서 shape와 빈도로 일어나는지 설명할 수 있는가?
- 이후 hybrid parallel topology를 볼 때 pipeline 축을 data/tensor/state-sharding 축과 분리해서 설명할 수 있는가?
