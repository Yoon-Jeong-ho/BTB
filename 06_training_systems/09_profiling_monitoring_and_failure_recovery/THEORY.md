# 09 Profiling, Monitoring, and Failure Recovery 이론 노트

## 핵심 개념

### 1. profiling intuition: training step은 time / memory / communication timeline이다
- profiling을 "느린 함수 하나를 찾는 작업"으로만 보면 large-model training에서 자주 길을 잃는다.
- 실제 학습 step은 보통 아래 요소가 겹쳐 움직인다.
  - data loading / host-side preparation
  - forward / backward compute
  - collective communication(all-reduce, all-gather, reduce-scatter, send/recv)
  - checkpoint save/load와 같은 I/O
  - logging, eval, synchronization boundary
- 따라서 profiling의 핵심 질문은 "어느 line이 느린가"보다 **step lifecycle 어디에서 대기가 생기고, 그 대기가 time / memory / communication 중 어느 축에 속하는가**에 가깝다.
- 같은 throughput 저하라도 원인은 완전히 다를 수 있다.
  - dataloader가 못 따라와 GPU가 기다리는 경우
  - matmul / attention kernel이 실제 compute hotspot인 경우
  - collective wait 때문에 rank들이 서로 기다리는 경우
  - checkpoint flush나 file system 병목이 step 경계에 끼어드는 경우
- 그래서 이 단위에서 profiling은 도구 이름을 외우는 것이 아니라, **학습 step을 해부해서 병목의 좌표를 잡는 사고법**으로 이해하는 편이 중요하다.

### 2. time profiling framing: 평균 step time만 보면 tail latency와 jitter를 놓치기 쉽다
- 운영에서 가장 흔한 실수는 평균 step time 하나만 보고 run 상태를 판단하는 것이다.
- 하지만 실제 문제는 다음처럼 나타날 수 있다.
  - 평균은 비슷하지만 특정 구간에서만 p95/p99 step time이 갑자기 늘어남
  - eval/save 직후만 step이 길어짐
  - 일부 rank만 느려져 전체 world가 기다림
  - warmup 이후는 빠른데, 긴 run 후반에만 점점 느려짐
- time profiling에서는 최소한 다음 분해가 필요하다.
  - **data wait**: 배치가 준비될 때까지 기다리는 시간
  - **compute**: forward/backward kernel이 실제 장치를 쓰는 시간
  - **communication wait**: 다른 rank와 collective를 맞추느라 기다리는 시간
  - **I/O / misc sync**: checkpoint, logging flush, host sync, eval 전환 등
- large-scale training에서는 특히 average보다 **jitter, tail, phase boundary**가 중요하다.
- 즉 "느리다"는 한 단어 대신, **언제부터, 어느 구간에서, 얼마나 흔들리기 시작했는가**를 시간축 위에 놓고 봐야 한다.

### 3. memory profiling framing: allocated / reserved / peak / lifetime을 함께 본다
- OOM이나 memory instability를 볼 때 단순 total memory 수치만 보면 충분하지 않다.
- 학습 step 안에는 여러 메모리 덩어리가 서로 다른 시점에 나타난다.
  - parameter / gradient / optimizer state
  - activation
  - temporary workspace / kernel scratch
  - communication buffer
  - caching allocator가 잡아 둔 reserved memory
- 특히 `allocated` 와 `reserved` 는 다른 의미를 가진다.
  - allocated: 지금 실제 tensor가 차지하는 메모리 감각에 가깝다.
  - reserved: allocator가 잡아 두고 있어 당장 OS/driver에 돌려주지 않은 영역까지 포함한다.
- 따라서 다음 질문이 중요하다.
  - peak memory는 step의 어느 시점에서 올라오는가?
  - eval/save/restart 직후에만 급증하는가?
  - allocated는 내려왔는데 reserved가 계속 높게 남는가?
  - 동일 batch인데 run 후반으로 갈수록 peak가 조금씩 커지는가?
- 이런 관찰은 단순 capacity 부족뿐 아니라 **fragmentation, unexpected tensor retention, checkpoint boundary memory spike** 같은 문제를 읽게 해 준다.

### 4. communication profiling framing: "느린 GPU"가 아니라 "기다리는 GPU"일 수 있다
- distributed training에서 GPU utilization이 낮다고 해서 항상 compute가 부족한 것은 아니다.
- 자주 보이는 실제 원인은 **다른 rank를 기다리는 통신 대기**다.
- collective communication은 본질적으로 참여자들이 어느 정도 맞춰 움직여야 하므로, 한 rank의 지연이 전체 step에 퍼질 수 있다.
- communication profiling에서 중요한 직관은 다음과 같다.
  - all-reduce / all-gather / reduce-scatter / send/recv는 모두 time cost를 가진다.
  - 문제는 통신 "양"뿐 아니라, **언제 그 통신이 step critical path에 들어오는가**다.
  - 한 rank가 dataloader stall이나 CPU preemption으로 늦어져도 나머지 rank는 collective boundary에서 함께 멈춰 보일 수 있다.
  - topology(NVLink, PCIe, intra-node vs cross-node) 차이는 같은 코드에서도 다른 병목 모양을 만든다.
- 그래서 communication 병목은 "네트워크가 느리다"는 일반론보다, **어느 collective가 어느 rank skew와 함께 critical path를 막았는가**로 읽는 편이 정확하다.

### 5. monitoring signals: profiler는 순간을, monitoring은 추세와 반복을 보여 준다
- profiler는 특정 window를 깊게 들여다보는 도구라면, monitoring은 긴 run에서 신호가 어떻게 drift하는지 보는 도구에 가깝다.
- 실전에서는 최소한 아래 계층을 함께 본다.

#### throughput / step-time signals
- samples/sec, tokens/sec, steps/sec
- step time p50 / p95 / p99
- warmup 이후 throughput 안정 여부
- eval/save/restart 이후 throughput 회복 여부

#### memory / device signals
- allocated / reserved / peak memory
- GPU utilization, SM occupancy 느낌, memory bandwidth pressure
- CPU RAM / page cache / host-side worker queue 상태
- per-rank imbalance 여부

#### optimization / numerical signals
- loss, validation metric
- grad norm, update norm
- NaN / inf 발생 여부
- mixed precision scaler 동작 이상 여부

#### liveness / recovery signals
- per-rank heartbeat
- last successful checkpoint timestamp
- checkpoint size / write duration / manifest consistency
- restart count, repeated failure interval

- 중요한 점은 단일 signal보다 **어떤 signal이 먼저 움직였는가**다.
- 예를 들어 throughput 저하보다 memory drift가 먼저 보였다면 leak/fragmentation을 의심할 수 있고, loss spike보다 grad norm spike가 먼저였다면 divergence framing이 더 자연스럽다.

### 6. failure triage: symptom을 먼저 분류해야 recovery도 정확해진다
학습 장애는 비슷해 보여도 대응 순서가 다르다. 먼저 어떤 계열 문제인지 나눠야 한다.

#### OOM(out-of-memory)
- 보통 비교적 명시적으로 터진다.
- 하지만 원인은 한 가지가 아니다.
  - batch/microbatch 자체가 큼
  - activation peak가 특정 구간에서만 높음
  - checkpoint/eval boundary에서 일시 spike 발생
  - fragmentation 때문에 reserved memory가 해제되지 않음
  - restart 후 world size/config가 달라져 memory layout이 바뀜
- 따라서 OOM triage는 "몇 GB 부족했는가"보다 **언제, 어떤 phase에서, allocated/reserved 중 무엇이 먼저 문제였는가**로 들어가야 한다.

#### hang / stall
- 프로세스가 죽지 않았는데 step이 끝나지 않거나, 로그가 멈춘 것처럼 보이는 상태다.
- 원인은 다양하다.
  - collective deadlock / timeout
  - dataloader stuck
  - 특정 rank만 느린 straggler 문제
  - checkpoint I/O가 오래 걸려 사실상 멈춘 것처럼 보임
  - 사용자 코드의 무한 대기 / barrier mismatch
- hang triage의 핵심은 **죽었는가, 기다리는가, 어디에서 기다리는가**를 분리하는 것이다.
- 그래서 heartbeat, per-rank last-log timestamp, in-flight collective 정보가 중요해진다.

#### divergence / instability
- loss spike, NaN, inf, grad norm explosion, resume 직후 metric 붕괴가 여기에 들어간다.
- 흔히 모델/optimizer 문제처럼 보이지만 실제로는 다음도 포함된다.
  - 잘못 복구된 optimizer/scheduler state
  - data corruption / sample ordering 변화
  - mixed precision scaler 이상
  - stale checkpoint에서 잘못 이어 붙인 state
- divergence triage는 "모델이 나쁘다"보다 **마지막 정상 step 이후 무엇이 달라졌는가**를 좁히는 작업이다.

#### storage / checkpoint / preemption failure
- checkpoint 파일 일부만 써졌거나, manifest가 불완전하거나, preemption 뒤 resume point가 어긋나는 경우다.
- large jobs에서는 이 문제가 단순 부가 이슈가 아니라, 실제 실험 지속 가능성을 좌우한다.
- 따라서 recovery는 checkpoint 존재 여부만이 아니라 **resume 가능성과 state consistency**까지 확인해야 한다.

### 7. checkpoint / restart / recovery concerns: recoverable run에는 저장 계약이 필요하다
- 복구 가능한 checkpoint는 보통 모델 파라미터만 저장했다고 끝나지 않는다.
- 최소한 다음을 어떤 형태로든 고려해야 한다.
  - model state
  - optimizer state
  - lr scheduler state
  - mixed precision scaler state
  - global step / epoch / consumed samples
  - dataloader or sampler position
  - RNG state
  - sharded/full checkpoint metadata manifest
  - world size / topology / config assumption
- 특히 distributed training에서는 "파일이 있다"와 "모든 rank가 같은 checkpoint identity를 기준으로 안전하게 resume한다"가 다른 문제다.
- recovery 관점에서 자주 보는 질문은 다음과 같다.
  - checkpoint write가 atomic했는가, 아니면 중간 실패로 partial file이 남았는가?
  - sharded checkpoint를 현재 world size/topology에서 바로 읽을 수 있는가?
  - resume 뒤 scheduler와 optimizer step count가 연속적인가?
  - sampler/data position이 복구되지 않아 일부 데이터를 중복 소비하거나 건너뛰지 않는가?
  - restart 직후 첫 수십 step이 이전 분포와 비슷하게 안정적인가?
- 따라서 recovery는 단순 재시작 버튼이 아니라, **last good checkpoint 선정 → state consistency 확인 → post-resume validation** 의 세 단계 계약으로 보는 편이 좋다.

### 8. OOM / hang / divergence debugging framing: 서로 다른 첫 질문을 던져야 한다
- OOM이 보이면 먼저 묻는다.
  - 어느 phase에서 터졌는가? (forward/backward/eval/save/resume)
  - allocated와 reserved 중 어떤 수치가 비정상적이었는가?
  - deterministic하게 재현되는가, 긴 run 후반에만 나타나는가?
- hang가 보이면 먼저 묻는다.
  - 모든 rank가 멈췄는가, 일부만 늦는가?
  - 마지막 로그/heartbeat는 어느 rank에서 끊겼는가?
  - collective boundary, dataloader, file I/O 중 어디에서 시간이 사라졌는가?
- divergence가 보이면 먼저 묻는다.
  - 마지막 정상 checkpoint/step은 어디였는가?
  - loss spike 이전에 grad norm, scaler, learning rate, data slice가 어떻게 움직였는가?
  - resume 직후라면 state continuity가 정말 맞는가?
- 즉 debugging framing은 증상을 한꺼번에 "실패"로 뭉뚱그리지 않고, **첫 질문을 다르게 던져 탐색 공간을 줄이는 작업**이다.

### 9. common confusion
- profiler 한 번 돌리면 monitoring이 필요 없다고 생각하는 실수
  - profiler는 깊고 짧게, monitoring은 얕고 길게 본다. 둘은 대체재가 아니다.
- GPU utilization이 높으면 잘 돌고 있다고 생각하는 실수
  - 높은 utilization 속에서도 잘못된 통신 대기나 divergence는 충분히 일어날 수 있다.
- checkpoint 파일이 남아 있으니 resume 가능하다고 생각하는 실수
  - partial write, stale manifest, missing optimizer/scheduler state면 복구 실패다.
- OOM을 무조건 batch size 문제로만 보는 실수
  - fragmentation, checkpoint boundary spike, offload/restart mismatch도 원인일 수 있다.
- hang와 slow step을 같은 말로 보는 실수
  - straggler, deadlock, timeout, long I/O는 관찰 패턴이 다르다.
- divergence를 순수 모델 품질 문제로만 보는 실수
  - 잘못된 resume, scaler 문제, corrupted batch, scheduler mismatch도 모두 divergence처럼 보일 수 있다.

## 무엇을 관찰할 것인가
- slowdown이 시작되기 직전 가장 먼저 움직인 signal은 무엇인가?
- peak memory는 어느 step / 어느 phase / 어느 rank에서 나타나는가?
- throughput 저하가 compute hotspot 때문인지 communication wait 때문인지 time breakdown으로 설명할 수 있는가?
- checkpoint 저장 시각, 크기, manifest, resume validation 결과를 서로 연결해서 볼 수 있는가?
- 장애 이후 첫 recovery 시도에서 무엇을 확인했고, 왜 그 checkpoint를 last good checkpoint로 간주했는가?
- 이후 frontier-style 실험에서 profiling snapshot과 monitoring trend, recovery log를 한 runbook으로 남길 수 있는가?
