# 04 FSDP, Checkpointing, and Offload 이론 노트

## 핵심 개념

### 1. FSDP sharding intuition: "모델 전체를 항상 모든 rank가 들고 있어야 하는가?"
- DDP에서는 각 rank가 모델 파라미터 복제본을 통째로 들고 있는 그림이 기본이다.
- ZeRO Stage 3도 parameter shard를 다루지만, 학습자가 체감하기에는 종종 "상태를 어떻게 나눌 것인가"에 초점이 먼저 온다.
- FSDP(Fully Sharded Data Parallel)는 이보다 더 직접적으로 **parameter/gradient/optimizer state를 shard된 기본 상태로 두고**, 필요한 순간에만 full parameter view를 잠깐 모아 쓰는 runtime이라는 감각이 중요하다.
- 그래서 FSDP를 "DDP에 옵션 몇 개 더 붙인 것"으로 보기보다, **모델 상태의 기본 거주 형태를 full replica에서 shard로 뒤집는 방식**으로 이해하는 편이 낫다.
- 질문은 늘 같다. "이 상태가 지금 꼭 GPU 위에 full form으로 있어야 하는가?" FSDP는 많은 경우 답을 "아니다, 직전에 모으고 끝나면 다시 쪼갠다"로 바꾼다.

### 2. parameter lifecycle: full parameter는 항상 있는 것이 아니라 잠깐 나타난다
- FSDP에서는 각 rank가 parameter shard만 들고 있다가 forward 계산 직전에 필요한 full parameter를 all-gather로 모은다고 이해하면 된다.
- forward가 끝나고 backward를 지나면 다시 shard 형태로 돌아가고, gradient/optimizer state도 shard-aware 흐름으로 관리된다.
- 즉 중요한 것은 "모델이 full인가 shard인가"를 고정 속성처럼 보는 것이 아니라, **언제 full view가 materialize되고 언제 해제되는가**를 보는 것이다.
- 이 runtime lifecycle을 모르면 다음과 같은 현상이 헷갈린다.
  - 왜 parameter 수는 같아도 steady-state memory와 peak memory가 다르게 보이는가?
  - 왜 어떤 step 구간에서만 memory spike가 생기는가?
  - 왜 wrap 정책이나 module granularity가 all-gather 패턴에 영향을 주는가?
- FSDP 실전 관찰에서는 평균 메모리보다도 **peak 순간이 어디서 생기는지**가 중요하다.

### 3. activation checkpointing motivation: 저장 대신 다시 계산한다
- large model에서 activation memory는 종종 parameter shard 절감만으로 해결되지 않는다.
- activation checkpointing은 forward 중간 결과를 모두 저장하지 않고, backward 때 필요한 구간을 다시 계산(recompute)하는 방식으로 메모리를 줄인다.
- 따라서 이 기법의 핵심 trade-off는 명확하다.
  - 얻는 것: activation 저장량 감소, 더 큰 sequence/batch/model 가능성
  - 잃는 것: 추가 forward 재계산, step time 증가, profiling 복잡도 증가
- 자주 생기는 오해는 "checkpointing = state save/load"라고 생각하는 것이다.
  - activation checkpointing은 **학습 중 메모리 절약 기법**이다.
  - model checkpoint save/load는 **학습 상태 저장/복구 기법**이다.
- 이 단위에서는 두 checkpointing을 모두 다루지만, 서로 다른 목적을 가진다는 점을 분리해서 기억해야 한다.

### 4. offload motivation: GPU 메모리를 비우는 대신 전송 시간을 산다
- FSDP만으로도 많은 메모리를 줄일 수 있지만, 여전히 optimizer state나 parameter shard 일부를 GPU 밖으로 내리고 싶을 때가 있다.
- CPU offload 또는 더 느린 저장장치 기반 offload의 직관은 단순하다.
  - GPU는 빠르지만 비싸고 좁다.
  - CPU/NVMe는 넓지만 느리다.
- 따라서 offload는 메모리를 공짜로 늘리는 마법이 아니라, **저장 위치를 바꾸어 GPU pressure를 낮추는 대신 transfer latency를 받아들이는 정책**이다.
- 실전에서는 다음 질문을 함께 본다.
  - peak GPU memory가 얼마나 줄었는가?
  - host-device transfer 때문에 step time이 얼마나 느려졌는가?
  - interconnect/NUMA/storage 차이로 variance가 커지지 않는가?
  - checkpoint resume이나 preemption 대응에 어떤 운영 상 이점/복잡도가 생기는가?
- 즉 offload의 목적은 "더 빨라지기"보다 **메모리 한계 안에서 학습을 성립시키기**에 가깝다.

### 5. memory-compute trade-offs: 무엇을 아끼면 무엇을 더 쓰게 되는가
- FSDP, activation checkpointing, offload를 함께 보면 공통 패턴이 있다. 메모리를 아끼면 보통 통신, 재계산, I/O 중 하나가 늘어난다.
- 대표적인 교환 관계는 다음과 같다.

#### FSDP sharding
- 절약: per-rank parameter/gradient/optimizer memory
- 비용: all-gather / reduce-scatter 통신 증가, runtime orchestration 복잡도 증가

#### activation checkpointing
- 절약: activation memory
- 비용: backward 시 추가 recomputation, step time 증가

#### CPU/NVMe offload
- 절약: GPU resident memory
- 비용: device-host 또는 storage I/O 대기, jitter 증가

- 따라서 좋은 설정은 "모든 절약 옵션을 다 켜기"가 아니라, **현재 모델·하드웨어·실험 목표에서 어떤 자원이 가장 부족한가**를 보고 결정해야 한다.
- 예를 들어 연구 iteration 속도가 중요하면 과한 offload가 불리할 수 있고, 일단 모델을 얹는 것이 더 중요하면 느리더라도 offload가 합리적일 수 있다.

### 6. full-state vs sharded-state loading: 저장 형식이 복구 전략을 바꾼다
- checkpoint를 저장할 때 가장 단순한 발상은 full state dict를 한곳에 모아 저장하는 것이다.
- 이 방식은 직관적이고 이식성이 좋다. inference export나 single-process debugging에도 유리하다.
- 하지만 large model에서는 full state를 한 번에 materialize하는 것 자체가 메모리 부담이 될 수 있다.
- 반대로 sharded state dict는 각 rank가 자기 shard 위주로 저장/복구하므로 대규모 학습에는 더 자연스럽다.
- 대신 다음 걱정이 생긴다.
  - world size가 달라져도 바로 복구되는가?
  - 다른 runtime/추론 환경으로 내보낼 때 추가 re-shard 또는 merge가 필요한가?
  - debugging할 때 "전체 모델 한 벌"을 쉽게 보는가?
- 따라서 full vs sharded의 선택은 파일 포맷 취향 문제가 아니라, **resume 대상 환경 / portability 요구 / memory budget**의 조합 문제다.
- 실전 감각으로는 다음처럼 정리할 수 있다.
  - full state: 단순하고 범용적이지만 무겁다.
  - sharded state: 대규모 학습에는 자연스럽지만 복구/이식성 계약을 더 명확히 관리해야 한다.

### 7. auto wrap, granularity, mixed precision을 따로 보지 말아야 한다
- FSDP에서는 어떤 모듈 경계로 shard/all-gather를 할지에 따라 통신 빈도와 peak memory가 달라진다.
- auto wrap policy는 이 granularity를 바꾸는 중요한 결정이다.
- mixed precision까지 함께 들어오면 gather되는 parameter dtype, optimizer state dtype, checkpoint export dtype 해석도 달라진다.
- 즉 FSDP 설정은 개별 체크박스의 나열이 아니라, **wrap granularity + precision + checkpointing + offload가 묶인 runtime 디자인**으로 읽어야 한다.
- 학습자가 자주 놓치는 포인트는 API 파라미터 이름보다도, 그 설정이 step timeline 어느 지점의 메모리/통신/복구 경로를 바꾸는가다.

### 8. common confusion
- FSDP를 "그냥 ZeRO Stage 3의 다른 이름"으로 생각하는 실수
  - 겹치는 intuition이 많지만, runtime wrapping, state dict handling, PyTorch native ecosystem 문맥에서 따로 읽어야 할 운영 감각이 있다.
- activation checkpointing과 model checkpoint 저장을 같은 뜻으로 쓰는 실수
  - 하나는 메모리 절약용 recomputation, 다른 하나는 save/load 복구다.
- shard됐으니 메모리 문제가 끝났다고 생각하는 실수
  - peak all-gather 순간, activation memory, optimizer state dtype, offload 비용이 남는다.
- offload를 켜면 더 큰 모델도 빠르게 학습된다고 기대하는 실수
  - 보통 목적은 속도 향상이 아니라 메모리 생존성 확보다.
- full state dict가 늘 더 안전하다고 생각하는 실수
  - 범용적이지만 큰 모델에서는 저장/로드 순간 자체가 병목이 될 수 있다.
- sharded checkpoint는 나중에 아무 데나 쉽게 옮길 수 있다고 생각하는 실수
  - world size 변화, runtime 차이, export 단계에서 추가 작업이 필요할 수 있다.

### 9. 이 단위에서 무엇을 관찰할 것인가
- FSDP wrap 후 steady-state memory와 forward/backward peak memory는 각각 어떻게 보이는가?
- activation checkpointing을 켰을 때 메모리 절감과 step time 증가가 어느 정도 교환되는가?
- CPU offload를 켰을 때 peak GPU memory는 얼마나 줄고, transfer 지연은 어떤 로그/프로파일 흔적으로 나타나는가?
- full state dict와 sharded state dict는 save, resume, export 경로에서 어떤 다른 제약을 만드는가?
- auto wrap granularity가 all-gather frequency와 peak memory를 어떻게 바꾸는가?
- 이후 tensor parallel이나 hybrid parallel을 볼 때, FSDP가 해결하는 문제와 해결하지 않는 문제를 어디서 구분할 수 있는가?
