# 05 Tensor Parallelism 이론 노트

## 핵심 개념

### 1. tensor parallelism intuition: "레이어 하나를 여러 장치가 함께 계산한다"
- data parallel은 보통 **같은 레이어를 여러 rank가 복제**해서 서로 다른 batch shard를 처리한다.
- ZeRO/FSDP류 sharding은 **상태를 나눠 저장**해 중복 메모리를 줄이는 쪽에 가깝다.
- tensor parallelism은 그와 달리, **레이어 내부의 실제 계산축 자체를 나눈다**.
- 즉 하나의 거대한 linear, attention projection, feed-forward expansion을 rank마다 일부씩 맡아 동시에 계산한다.
- 그래서 tensor parallelism은 "모델을 여러 장치에 배치한다"는 넓은 의미의 model parallelism 안에서도, 특히 **intra-layer parallelism**이라고 보는 편이 정확하다.

### 2. 왜 필요한가: 큰 matmul은 레이어 내부에서 이미 병목이 된다
- Transformer block에서는 hidden size와 intermediate size가 커질수록 QKV projection, output projection, MLP up/down projection이 매우 큰 행렬곱이 된다.
- 이때 병목은 batch만 커서 생기지 않는다. **레이어 하나의 weight / activation / temporary buffer** 자체가 한 GPU에 부담이 될 수 있다.
- 따라서 어떤 경우에는 모델 상태를 shard하는 것만으로 부족하고, **forward/backward가 진행되는 동안 활성 계산 자체를 여러 장치로 찢어야** 한다.
- tensor parallelism은 이 문제를 "레이어 바깥에서 나누기"가 아니라 **레이어 안에서 나누기**로 푼다.
- large language model 학습에서 tensor parallel이 자주 등장하는 이유도, hidden dimension과 feed-forward expansion이 커질수록 intra-layer split이 자연스러운 선택지가 되기 때문이다.

### 3. matrix split 기본 감각: column parallel과 row parallel
하나의 linear layer를 생각하면 tensor parallel intuition이 가장 쉽게 잡힌다.

#### column parallel linear
- weight의 output dimension 쪽을 여러 rank로 나눈다.
- 각 rank는 전체 출력 feature 중 일부만 계산한다.
- 따라서 각 rank는 **partial output activation**을 만든다.
- 다음 연산이 shard된 출력 상태를 그대로 소비할 수 있으면 all-gather를 미룰 수 있고, full output이 필요하면 다시 모아야 한다.
- 직관적으로는 "출력 채널을 나눠 계산한다"에 가깝다.

#### row parallel linear
- weight의 input dimension 쪽을 여러 rank로 나눈다.
- 보통 입력 activation도 그에 맞게 shard되어 들어온다.
- 각 rank는 자기 shard에 대한 partial contribution만 계산하므로, 최종 output을 만들려면 **partial sum을 합치는 collective**가 필요하다.
- 이때 all-reduce나 reduce-scatter 같은 연산이 핵심이 된다.
- 직관적으로는 "입력 feature 축을 나눠 각 rank가 일부 내적만 계산한 뒤 결과를 합친다"라고 보면 된다.

### 4. activation도 함께 쪼개서 봐야 한다
- tensor parallelism은 weight shard만 나누는 문제가 아니다.
- 레이어 출력이 다음 레이어 입력으로 이어지므로, **activation을 어떤 shape로 어떤 rank가 쥘 것인가**가 매우 중요하다.
- 예를 들어 column-parallel 결과를 full tensor로 바로 모아 버리면 구현은 단순해지지만 통신량이 늘 수 있다.
- 반대로 shard된 activation을 가능한 오래 유지하면 메모리와 bandwidth를 아낄 수 있지만, 다음 레이어도 그 shard layout을 이해해야 한다.
- 그래서 tensor parallel 설계는 "행렬을 어떻게 나눌까"만이 아니라 **activation layout contract를 몇 레이어 연속으로 유지할 수 있는가**까지 포함한다.

### 5. attention / feed-forward에서 왜 잘 맞는가
- multi-head attention은 head 축이 상대적으로 자연스러운 분할점이다.
- 예를 들어 전체 head 수가 32이고 tensor parallel world size가 4면, 각 rank가 8개 head를 맡는 식의 사고가 가능하다.
- feed-forward network도 hidden-to-intermediate 확장 차원이 매우 크므로 column/row parallel linear 패턴을 적용하기 좋다.
- 그래서 tensor parallelism은 특히 Transformer block에서 반복 가능성이 높다.
- 즉 "한 번만 쓰는 트릭"이 아니라, **attention projection과 MLP projection을 따라 반복적으로 나타나는 layout 전략**으로 이해해야 한다.

### 6. communication overhead: 메모리 절감은 공짜가 아니다
- tensor parallelism은 각 rank의 weight/activation 메모리 부담을 줄여 준다.
- 대신 레이어 안에서 계산한 partial result를 합치거나 재배치해야 하므로, **collective communication이 거의 매 블록마다 눈에 띄게 등장**한다.
- 중요한 질문은 항상 두 가지다.
  - 얼마나 큰 메모리/계산 부담을 줄였는가?
  - 그 대가로 얼마나 자주, 얼마나 큰 tensor를 통신하는가?
- interconnect가 충분히 빠르지 않으면 matmul 계산보다 all-gather / reduce-scatter / all-reduce 대기가 더 크게 보일 수 있다.
- 그래서 tensor parallelism은 대개 **고대역폭 intra-node 연결(NVLink 등)** 안에서 우선 적용되고, 느린 링크를 넘는 cross-node tensor parallel은 더 신중하게 다뤄진다.

### 7. latency vs throughput trade-off를 읽는 법
- 장치가 늘면 단순 기대는 "더 빨라질 것"이지만, tensor parallel에서는 그렇지 않을 수 있다.
- per-rank matmul 크기가 너무 작아지면 GPU utilization이 떨어질 수 있다.
- 반대로 tensor shard가 너무 크면 메모리 절감 효과가 약해진다.
- 결국 좋은 설정은 "GPU 수를 최대한 많이 쓴다"가 아니라, **레이어 계산량과 통신량의 비율이 균형을 이루는 shard 크기**를 찾는 일이다.
- 실전에서는 hidden size, sequence length, batch, interconnect, kernel fusion 여부가 모두 latency/throughput 균형에 영향을 준다.

### 8. sharding 접근과의 관계: ZeRO/FSDP와 무엇이 다른가
- ZeRO/FSDP는 주로 **상태를 어떻게 저장하고 필요할 때 어떻게 모을 것인가**에 초점을 둔다.
- tensor parallelism은 **실제 active computation 자체를 rank별로 분담**한다.
- 둘 다 memory pressure를 줄이는 데 쓰일 수 있지만, 줄이는 방식이 다르다.
  - ZeRO/FSDP: full layer semantics를 유지하면서 상태 복제 비용을 줄임
  - Tensor parallel: layer computation graph 내부를 분할해서 각 rank가 partial compute를 수행
- 그래서 대형 모델에서는 둘이 경쟁 관계라기보다 **서로 다른 축을 담당하는 조합 가능 전략**이 된다.
- later hybrid parallel에서는 같은 run 안에 data parallel + sharding + tensor parallel이 함께 들어가는 이유가 바로 여기에 있다.

### 9. pipeline parallel과의 관계: 레이어 안을 나누는가, 레이어 사이를 나누는가
- tensor parallel은 **한 레이어 내부**를 동시에 여러 장치가 계산한다.
- pipeline parallel은 **레이어 묶음(스테이지)** 을 서로 다른 장치가 순차적으로 맡는다.
- tensor parallel의 대표 질문은 "이 matmul을 어느 차원으로 나눌까?"이고,
- pipeline parallel의 대표 질문은 "이 레이어 구간을 어느 stage 경계에서 자를까?"다.
- 둘은 분리된 개념이므로 later hybrid setup에서는 같은 모델 안에서 동시에 쓰일 수 있다.

### 10. common confusion
- tensor parallelism을 그냥 "GPU를 더 많이 쓰는 data parallel"로 오해하는 실수
  - 핵심은 batch shard가 아니라 **레이어 내부 compute shard**다.
- row parallel / column parallel 이름을 weight layout 표기와 혼동하는 실수
  - 프레임워크마다 텐서 저장 방향이 달라 보여도, 중요한 것은 **어느 feature 차원을 나누는가**다.
- tensor parallel이면 communication이 줄어든다고 생각하는 실수
  - 보통 메모리는 줄지만 layer-level collective는 늘어난다.
- attention head를 나누면 나머지 projection도 자동으로 쉬워진다고 생각하는 실수
  - 실제로는 projection 경계마다 activation layout과 collectives를 같이 봐야 한다.
- tensor parallel만 있으면 sharding/pipeline이 불필요해진다고 생각하는 실수
  - large-scale training에서는 memory, bandwidth, latency, topology 문제를 한 축만으로 풀기 어려워 조합이 흔하다.

## 무엇을 관찰할 것인가
- 각 rank가 어떤 weight shard shape와 activation shard shape를 들고 있는가?
- column parallel과 row parallel에서 full tensor를 다시 모으는 시점이 어디인가?
- attention/MLP block에서 collective communication이 블록당 몇 번 정도 발생하는가?
- memory 절감이 커질수록 step time에서 latency 대기가 얼마나 보이는가?
- 같은 tensor parallel world size라도 interconnect 품질에 따라 throughput 체감이 어떻게 달라지는가?
- ZeRO/FSDP, pipeline parallel, hybrid parallel을 볼 때 tensor parallel이 어떤 독립 축으로 작동하는가?
