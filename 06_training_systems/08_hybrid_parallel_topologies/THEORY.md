# 08 Hybrid Parallel Topologies 이론 노트

## 핵심 개념

### 1. 왜 hybrid parallel이 필요한가: 한 가지 축만으로는 큰 모델을 감당하기 어렵다
- small-to-medium scale에서는 data parallel만으로도 충분해 보일 수 있다.
- 하지만 모델이 커질수록 병목은 한 종류가 아니라 여러 층으로 겹친다.
  - parameter / optimizer state가 너무 커서 메모리에 안 들어간다.
  - activation이 길어진 sequence와 microbatch 때문에 커진다.
  - 한 레이어 내부 matmul이 너무 커서 single device compute / memory 한계를 넘는다.
  - stage를 나누지 않으면 모델 전체 레이어 스택을 한 장치가 감당하기 어렵다.
- 그래서 현실의 large-model training은 보통 data parallel, tensor parallel, pipeline parallel, FSDP/ZeRO류 sharding을 함께 조합한다.
- hybrid parallel topology의 핵심은 "기법을 많이 아는가"가 아니라, **각 기법이 해결하는 병목을 서로 다른 축으로 구분한 뒤 조합하는가**다.

### 2. 네 가지 축을 무엇을 나누는가로 다시 정리하기

#### data parallel
- 같은 모델 복제본을 여러 rank가 들고 서로 다른 mini-batch를 처리한다.
- 주로 throughput과 global batch 확장에 기여한다.
- gradient synchronization(all-reduce)이 핵심 통신으로 등장한다.

#### tensor parallel
- 한 레이어 내부의 큰 linear/attention 연산을 여러 rank가 함께 계산한다.
- 모델 상태 메모리 일부를 줄이고, intra-layer compute를 분산한다.
- row/column parallel linear, attention head split, collective communication이 핵심이다.

#### pipeline parallel
- 레이어 스택을 stage로 나누고 microbatch를 시간축으로 흘린다.
- 모델 전체를 stage별로 나눠 적재할 수 있게 해 주지만 bubble과 load imbalance를 만든다.
- activation send/recv, warmup/steady-state/cooldown이 중요하다.

#### FSDP / ZeRO류 state sharding
- parameter / gradient / optimizer state를 shard된 기본 상태로 둔다.
- per-rank resident memory를 줄이는 대신 all-gather / reduce-scatter / state-dict handling 복잡도가 늘어난다.
- 주 관심사는 compute 분할보다도 **상태의 거주 위치와 lifecycle**이다.

- hybrid topology를 설계할 때 가장 먼저 해야 할 일은 이 축들을 "다 비슷한 분산 기법"으로 뭉개지 않고, **무엇을 나누는 축인지** 기준으로 다시 세우는 것이다.

### 3. topology design intuition: 모델과 하드웨어를 동시에 본다
- 좋은 hybrid topology는 모델 관점만으로도, 하드웨어 관점만으로도 결정되지 않는다.
- 일반적인 사고 순서는 다음과 같다.
  1. 모델 규모와 학습 목표를 정한다. (`params`, `sequence length`, `target tokens/step`, `global batch`)
  2. 어떤 메모리 항목이 먼저 터지는지 확인한다. (parameter / optimizer / activation)
  3. 어떤 통신이 가장 자주, 가장 크게 일어날지 가늠한다.
  4. 클러스터 링크 구조를 본다. (node 내부 빠른 링크 vs node 간 느린 링크)
  5. 통신이 민감한 축을 더 빠른 링크 위에, 상대적으로 덜 민감한 축을 바깥쪽에 배치한다.
- 자주 쓰이는 직관은 다음과 같다.
  - tensor parallel은 intra-layer collective가 잦으므로 보통 node 내부의 빠른 링크 위에 두는 편이 유리하다.
  - pipeline parallel은 activation 경계만 잘 정리되면 node 간으로도 배치할 여지가 있다.
  - data parallel은 비교적 바깥 축에 두기 쉽지만, gradient all-reduce 규모가 커지면 역시 네트워크 제약을 받는다.
  - FSDP는 memory-saving 목적이 강하므로 data parallel group 안쪽/바깥쪽 어디에 놓을지에 따라 checkpoint·all-gather·state layout 감각이 달라진다.
- 결국 topology 설계는 **parallel axis를 수학적으로 곱해서 world size를 맞추는 일**이 아니라, 어느 축을 어느 링크 계층에 올릴지를 정하는 일이다.

### 4. communication trade-offs: 축을 섞으면 통신도 합쳐진다
- hybrid setup에서 통신은 하나만 존재하지 않는다.
- 축별 대표 통신을 분리하면 다음과 같다.

#### data parallel communication
- gradient all-reduce / reduce-scatter
- optimizer-step 전후의 replica synchronization
- global batch가 커질수록 동기화 볼륨과 cadence가 중요해진다.

#### tensor parallel communication
- row/column parallel linear 이후의 all-reduce / all-gather
- attention head / MLP shard 사이 partial output 결합
- latency와 bandwidth 모두에 민감하며, 느린 링크를 타면 급격히 비싸진다.

#### pipeline parallel communication
- stage boundary activation send/recv
- forward/backward 흐름의 순서 의존성
- microbatch 수와 stage imbalance가 throughput을 좌우한다.

#### FSDP / state sharding communication
- parameter all-gather, gradient reduce-scatter, checkpoint state re-materialization
- wrap granularity와 checkpoint strategy에 따라 peak pattern이 달라진다.

- hybrid topology에서 중요한 것은 이 통신들이 **서로 독립적으로 사라지지 않는다**는 점이다.
- 예를 들어 tensor parallel과 FSDP를 함께 쓰면, intra-layer collectives와 shard lifecycle collectives가 같은 step timeline에 겹칠 수 있다.
- pipeline까지 함께 들어오면 특정 stage 경계에서 activation transfer가 몰리고, data parallel sync까지 outer loop에서 붙는다.
- 따라서 통신 최적화는 "한 축의 collective만 빠르게 만들기"보다, **어떤 축의 통신이 어느 링크에서 겹치는가**를 보는 문제다.

### 5. memory trade-offs: 메모리를 줄이는 축과 새 메모리를 만드는 축을 함께 본다
- hybrid parallel을 쓰는 이유 중 하나는 memory fit이지만, 모든 축이 똑같은 방식으로 메모리를 줄여 주는 것은 아니다.

#### memory를 직접 줄이는 축
- FSDP / ZeRO류: parameter / gradient / optimizer state resident footprint를 줄인다.
- tensor parallel: per-rank weight shard로 일부 layer-local memory를 줄인다.
- pipeline parallel: stage별로 레이어를 나눠 한 rank가 들고 있는 전체 모델 구간을 줄인다.

#### memory를 새로 복잡하게 만드는 축
- pipeline parallel: in-flight microbatch 수와 schedule에 따라 activation 보관 패턴이 복잡해진다.
- tensor parallel: partial activation / gathered output 버퍼가 추가로 보일 수 있다.
- FSDP: all-gather 순간의 temporary full-parameter view가 peak memory를 만든다.
- data parallel + grad accumulation: global batch를 키우는 방향으로 쓰면 optimizer cadence와 activation persistence 해석이 달라진다.

- 따라서 topology를 고를 때는 "메모리를 가장 많이 줄이는 축"만 볼 것이 아니라, **steady-state resident memory와 step 중 peak memory를 각각 따로 봐야 한다**.
- 실전에서는 이 distinction을 놓쳐서 "이론상 shard됐는데 왜 아직 OOM이 나는가" 같은 혼란이 자주 생긴다.

### 6. 모델 규모를 하드웨어 배치로 옮기는 직관
- hybrid design은 추상 개념을 실제 하드웨어 좌표로 내리는 과정이다.
- 다음 질문 순서가 유용하다.

#### 질문 1: 지금 문제는 memory fit인가 throughput target인가?
- memory fit이 먼저면 FSDP / tensor / pipeline으로 resident footprint를 줄이는 축을 먼저 본다.
- throughput target이 먼저면 data parallel 확대, pipeline fill, communication overlap이 더 중요해질 수 있다.

#### 질문 2: 모델의 "너무 큰 부분"은 어디인가?
- 전체 parameter state가 큰가?
- 일부 layer matmul이 너무 큰가?
- sequence length 때문에 activation이 폭증하는가?
- 레이어 깊이가 너무 커서 stage 분할이 필요한가?

#### 질문 3: 클러스터 링크 구조는 어떤가?
- 단일 노드의 NVLink/NVSwitch는 빠르지만, node 간 InfiniBand/Ethernet은 상대적으로 느릴 수 있다.
- 통신 빈도가 높은 tensor parallel을 node 간에 걸치면 큰 비용을 치를 수 있다.
- pipeline parallel은 stage 경계 activation만 관리하면 node 간 배치 여지가 더 있다.

#### 질문 4: checkpoint / restart / 운영 복잡도는 감당 가능한가?
- topology가 복잡할수록 checkpoint save/load 계약, rank mapping, recovery 절차도 복잡해진다.
- 단순히 fit만 되는 topology보다, **운영 가능한 topology**가 더 좋은 선택일 수 있다.

- 예를 들어 70B급 모델을 64 GPU에 올릴 때는 `TP x PP x DP x FSDP` 조합을 후보로 놓고,
  - TP는 node 내부로 묶고
  - PP는 stage 균형이 맞도록 node 사이를 자르고
  - DP/FSDP는 outer replica / shard 축으로 두는 식의 사고를 하게 된다.
- 이때 중요한 것은 정답 수식보다 **병목의 위치를 논리적으로 추적할 수 있는가**다.

### 7. common confusion
- hybrid parallel을 "각 parallelism을 다 켠 상태"로 이해하는 실수
  - 실제로는 병목에 맞는 축만 조합해야 하며, 필요 없는 축을 넣으면 통신과 운영 복잡도만 늘어난다.
- data parallel과 FSDP를 같은 축으로 뭉개는 실수
  - 둘 다 replica 그룹 문맥에 보일 수 있지만, 하나는 모델 복제 기반 throughput 확장, 다른 하나는 state sharding 기반 memory 절감에 더 가깝다.
- tensor parallel을 아무 링크 위에나 올려도 된다고 생각하는 실수
  - intra-layer collective가 잦기 때문에 느린 링크를 타면 step time이 급격히 나빠질 수 있다.
- pipeline stage를 레이어 수만 맞춰 나누면 충분하다고 생각하는 실수
  - 실제로는 stage별 compute time, activation size, skip/residual 구조, optimizer cadence까지 봐야 한다.
- world size를 축의 곱으로만 맞추면 topology 설계가 끝났다고 생각하는 실수
  - 같은 `TP=4, PP=2, DP=8`이라도 어떤 축이 node 안/밖에 놓이는지에 따라 성능과 안정성이 크게 달라진다.
- memory fit만 되면 좋은 topology라고 생각하는 실수
  - throughput, recovery complexity, checkpoint portability, failure blast radius도 함께 봐야 한다.

### 8. 이 단위에서 특히 관찰할 포인트
- 특정 topology 후보에서 가장 잦은 collective는 무엇이며, 어떤 링크 계층을 타는가?
- tensor parallel, pipeline parallel, FSDP가 함께 있을 때 peak memory가 어느 시점에서 생기는가?
- global batch를 맞추기 위해 grad accumulation을 늘렸을 때 pipeline fill / optimizer cadence는 어떻게 달라지는가?
- stage imbalance와 inter-node communication 중 무엇이 throughput 병목을 더 크게 만드는가?
- checkpoint 저장/복구 시 topology 정보가 어떤 메타데이터 계약으로 남아야 하는가?
- profiling을 시작하면 어떤 counter/log부터 봐야 topology misalignment를 빨리 잡을 수 있는가?

### 9. 이 단위가 남기는 감각
- hybrid parallel topology는 "분산 기술 목록"이 아니라, **모델-하드웨어-운영 제약을 동시에 푸는 배치 설계 문제**다.
- 좋은 학습자는 각 parallel axis를 따로 외우는 대신, 다음 문장으로 요약할 수 있어야 한다.
  - data parallel은 replica와 batch 축을 다룬다.
  - tensor parallel은 layer 내부 compute 축을 다룬다.
  - pipeline parallel은 stage와 시간축 실행을 다룬다.
  - FSDP/ZeRO는 state 거주 위치와 memory lifecycle을 다룬다.
- 이 구분이 선명해지면 이후 profiling/failure recovery를 볼 때도, 문제가 어느 축의 설계에서 생겼는지 훨씬 빠르게 역추적할 수 있다.
