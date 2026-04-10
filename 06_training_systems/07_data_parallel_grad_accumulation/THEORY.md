# 07 Data Parallel + Grad Accumulation 이론 노트

## 핵심 개념

### 1. data parallel intuition: 같은 모델을 여러 rank가 복제해서 서로 다른 shard를 본다
- data parallel의 기본 감각은 **모델은 복제되고 데이터가 나뉜다**는 것이다.
- 각 rank는 같은 파라미터로 forward/backward를 수행하지만, 서로 다른 mini-batch shard를 처리한다.
- backward 뒤에는 rank별 gradient를 동기화해 같은 업데이트 방향을 맞춘다.
- 그래서 data parallel의 핵심 질문은 "모델을 어떻게 쪼갤까"보다, **각 rank가 어떤 데이터를 보고 언제 gradient를 맞출까**에 가깝다.
- DDP는 이 축의 대표 구현이며, large-scale training에서 가장 기본이 되는 병렬화 기준선이다.

### 2. effective batch size: local / global / effective를 분리해 봐야 한다
- single-process에서는 batch size 하나만 생각해도 되는 경우가 많다.
- data parallel이 들어오면 적어도 세 가지 숫자를 따로 봐야 한다.
  - local batch size: 각 rank가 한 microstep에 직접 처리하는 샘플 수
  - global batch per microstep: `local_batch × world_size`
  - effective batch per optimizer step: `local_batch × world_size × grad_accum_steps`
- 예를 들어 local batch 8, world size 4, accumulation 4면
  - global batch per microstep = 32
  - effective batch per optimizer step = 128
- 이 구분이 무너지면 실험 로그를 읽을 때 "배치가 128"이 local인지 effective인지 헷갈리게 되고, optimizer dynamics 해석도 함께 틀어지기 쉽다.

### 3. grad accumulation intuition: step을 늦춰 큰 batch를 흉내 낸다
- grad accumulation은 매 microstep마다 optimizer를 갱신하지 않고, 여러 번의 forward/backward 결과를 gradient buffer에 쌓아 두었다가 나중에 한 번 step하는 방식이다.
- 직관적으로는 **메모리는 작은 batch처럼 쓰되, optimizer는 더 큰 batch를 본 것처럼 행동하게 만드는 우회로**다.
- 특히 local batch를 더 키우면 OOM이 나는 상황에서 accumulation은 effective batch를 늘릴 수 있는 가장 흔한 방법이다.
- 다만 accumulation은 "공짜 batch enlargement"가 아니다.
  - forward/backward 횟수는 실제로 더 많이 필요하다.
  - optimizer step 빈도는 줄어든다.
  - logging / lr schedule / gradient clipping 시점도 accumulation boundary에 맞춰 다시 해석해야 한다.
- 그래서 accumulation은 단순 메모리 트릭이 아니라 **step cadence를 바꾸는 운영 정책**으로 보는 편이 정확하다.

### 4. sync vs accumulation trade-off: 언제 통신하고 언제 미루는가
- vanilla DDP 직관에서는 각 microstep의 backward 뒤에 gradient synchronization이 일어난다고 생각하면 된다.
- 그런데 accumulation을 넣으면, 모든 microstep마다 sync할 수도 있고 accumulation boundary에서만 sync하도록 미룰 수도 있다.
- boundary까지 sync를 미루면 장점이 있다.
  - communication 횟수가 줄어든다.
  - 작은 microbatch마다 all-reduce하는 오버헤드를 줄일 수 있다.
- 하지만 trade-off도 생긴다.
  - optimizer step 한 번이 늦어진다.
  - gradient가 더 오래 버퍼에 남아 있으므로 debugging 관찰이 복잡해진다.
  - 잘못 구현하면 loss normalization / clipping / scheduler step 타이밍이 틀어지기 쉽다.
- 따라서 accumulation에서 중요한 질문은 "몇 step 쌓을까"만이 아니라, **그 기간 동안 synchronization과 optimizer step을 어떤 규칙으로 묶을까**다.

### 5. throughput vs memory balance: local batch를 키우는 것과 accumulation을 늘리는 것은 다르다
- local batch를 키우면 GPU 한 번의 matmul/kernel이 더 큰 텐서를 처리하므로 hardware utilization이 좋아질 수 있다.
- 하지만 local batch를 키우는 순간 activation memory도 함께 커져 OOM 위험이 높아진다.
- accumulation을 늘리면 local microbatch는 작게 유지할 수 있어 memory ceiling은 낮아진다.
- 대신 effective batch 하나를 만들기 위해 여러 번 forward/backward를 반복해야 하므로, wall-clock step latency는 길어질 수 있다.
- 즉 다음 두 접근은 비슷한 effective batch를 만들더라도 시스템 흔적이 다르다.
  - 큰 local batch + 적은 accumulation
  - 작은 local batch + 많은 accumulation
- 실전에서는 memory fit, GPU utilization, communication frequency, optimizer noise를 함께 보고 균형점을 찾는다.

### 6. optimizer dynamics 관점: 같은 effective batch여도 완전히 같은 run은 아니다
- 흔히 accumulation을 쓰면 "큰 batch와 완전히 같다"고 쉽게 말하지만, 실제 관찰은 더 조심해야 한다.
- effective batch가 같아도 다음 요소들이 다르면 run의 느낌이 달라질 수 있다.
  - local microbatch에서의 kernel efficiency
  - mixed precision / loss scaling 동작
  - gradient clipping 시점
  - scheduler step을 microstep 기준으로 둘지 optimizer-step 기준으로 둘지
  - logging frequency와 metric smoothing 기준
- 즉 accumulation은 optimizer가 보는 gradient aggregate를 크게 만들 수는 있지만, **runtime과 운영 계측까지 완전히 동일하게 만들지는 않는다**.
- 그래서 이 단위에서는 "수식상 batch가 같다"와 "실행 흔적이 같다"를 분리해서 보는 태도가 중요하다.

### 7. common confusion
- data parallel과 grad accumulation을 같은 종류의 기법으로 보는 실수
  - data parallel은 **동시에 더 많은 데이터를 처리하는 병렬화 축**이고, accumulation은 **step을 늦추는 스케줄링 축**이다.
- global batch와 effective batch를 혼동하는 실수
  - accumulation이 있으면 microstep 기준 global batch와 optimizer-step 기준 effective batch가 달라진다.
- accumulation만 늘리면 throughput도 자동으로 좋아진다고 생각하는 실수
  - memory는 덜 쓰더라도 forward/backward 반복이 늘어 step latency가 길어질 수 있다.
- DDP에서 accumulation을 쓰면 communication이 완전히 사라진다고 생각하는 실수
  - 보통은 sync 시점이 늦춰질 뿐이며, 경계에서는 여전히 gradient contract를 맞춰야 한다.
- loss를 accumulation step 수로 나누지 않아 gradient scale이 달라지는 실수
  - 구현에서는 normalization과 clipping, scheduler step 시점을 함께 점검해야 한다.
- local batch를 지나치게 작게 두고 accumulation만 크게 늘려 GPU utilization을 망치는 실수
  - memory는 안전해져도 runtime efficiency는 나빠질 수 있다.

### 8. 무엇을 관찰할 것인가
- local batch, world size, accumulation steps가 바뀔 때 global/effective batch 계산이 어떻게 달라지는가?
- optimizer step이 실제로 몇 microstep마다 한 번 일어나는지 trace로 설명할 수 있는가?
- accumulation boundary 전후로 gradient synchronization이 언제 발생하는지 명확히 말할 수 있는가?
- 더 큰 local batch와 더 많은 accumulation 중 무엇이 memory ceiling을 더 직접적으로 건드리는가?
- 같은 effective batch여도 throughput, latency, optimizer noise가 왜 다르게 체감될 수 있는가?
- 이후 hybrid parallel topology를 볼 때 data-parallel 축이 batch budget을 담당한다는 사실을 분리해서 설명할 수 있는가?
