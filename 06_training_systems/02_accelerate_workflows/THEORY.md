# 02 Accelerate Workflows 이론 노트

## 핵심 개념

### 1. Accelerate abstraction intuition: 같은 PyTorch loop를 여러 실행 환경으로 옮기는 적응 계층
- Accelerate의 핵심 직관은 "새로운 training framework를 하나 더 배운다"가 아니라, **기존 PyTorch 학습 루프를 다양한 device/distributed 환경에 맞게 적응시키는 공통 인터페이스를 둔다** 에 가깝다.
- 공식 quicktour가 보여 주는 최소 패턴도 이 점에 맞춰져 있다. `Accelerator()`를 만들고, `prepare(...)`로 model/optimizer/dataloader/scheduler를 넘기고, `backward(loss)`를 통해 backward 경로를 맞춘다.
- 즉 Accelerate는 training objective나 model architecture를 바꾸는 도구가 아니라, **실행 환경 차이 때문에 생기는 주변 보일러플레이트를 줄이는 orchestration layer** 로 보는 편이 정확하다.
- 그래서 이 단위의 질문은 "Accelerate가 강력한가?"보다, **어떤 종류의 복잡도를 흡수하고 어떤 복잡도는 그대로 남기는가?** 여야 한다.

### 2. Accelerator 객체는 무엇을 알고 있는가
- `Accelerator`는 스크립트가 어떤 환경에서 실행됐는지 보고 현재 distributed setup을 해석한다.
- 공식 문서 기준으로 여기에는 `device`, `distributed_type`, `local_process_index`, `process_index`, `num_processes`, `mixed_precision`, `sync_gradients` 같은 상태가 포함된다.
- 내부 메커니즘 문서는 이 정보를 `AcceleratorState`가 들고 있으며, 최초 `Accelerator()` 생성 시 환경 분석과 setup 초기화가 이뤄진다고 설명한다.
- 직관적으로 보면 `Accelerator`는 단순 유틸 함수 묶음이 아니라, **이 프로세스가 전체 학습 시스템 안에서 어디에 있는지 아는 상태 객체** 다.
- 그래서 Accelerate를 쓰더라도 rank/world size/local rank 개념이 완전히 사라지는 것은 아니다. 다만 평소 코드는 그 값을 직접 분기하지 않아도 되는 경우가 많아진다.

### 3. device placement intuition: `.cuda()`를 지워도 device 개념이 사라지는 것은 아니다
- Quicktour와 migration 문서는 `.cuda()` 혹은 `tensor.to(device)` 호출을 지우고 Accelerate가 자동 device placement를 맡게 두는 패턴을 권장한다.
- `prepare(...)` 이후 dataloader가 내놓는 batch도 적절한 device로 옮겨질 수 있고, model/optimizer도 준비 과정에서 해당 backend에 맞는 장치 상태로 감싸진다.
- 이 단순화가 중요한 이유는 코드 곳곳의 수동 device handling을 줄여 **단일 GPU -> multi-GPU -> TPU** 이동 시 코드 수정량을 줄이기 때문이다.
- 하지만 이것이 device 개념 자체를 지워 주는 것은 아니다.
  - optimizer를 model보다 먼저 잘못된 device에서 만들면 여전히 문제가 날 수 있다.
  - `device_placement=False`로 끄면 다시 사용자가 수동 배치를 책임져야 한다.
  - gather/save/checkpoint/load 시점에는 어떤 장치와 precision에서 상태가 오가는지 이해가 필요하다.
- 즉 Accelerate는 **device bookkeeping의 반복 작업** 을 줄여 주지만, device semantics 자체를 학습할 필요까지 없애 주지는 않는다.

### 4. `prepare(...)`는 실제로 무엇을 바꾸는가
- 공식 internal mechanism 설명에 따르면 `prepare(...)`는 다음을 수행한다.
  - model을 현재 distributed setup에 맞는 container/wrapper로 감싼다.
  - optimizer를 `AcceleratedOptimizer` 계열 wrapper로 감싼다.
  - scheduler를 `AcceleratedScheduler` 계열 wrapper로 감싼다.
  - dataloader를 sharding/device placement가 가능한 새 객체로 다시 만든다.
- 여기서 가장 중요한 직관은 `prepare(...)`가 단순히 "device로 옮기는 함수"가 아니라, **훈련에 참여하는 핵심 객체들의 실행 계약을 backend-aware 형태로 재조립하는 단계** 라는 점이다.
- 특히 dataloader는 process별 shard, RNG 동기화, batch device 이동 같은 동작을 추가해야 하므로 wrapper를 넘어서 아예 새 dataloader 형태가 만들어질 수 있다.
- 따라서 관찰 포인트는 다음과 같다.
  - batch가 process별로 어떻게 분배되는가?
  - shuffling/randomness는 각 프로세스에서 어떻게 맞춰지는가?
  - model wrapper가 DDP/FSDP/DeepSpeed 등의 어떤 backend와 연결되는가?
- 이 질문을 이해해야만 이후 backend-specific unit을 볼 때 Accelerate와 실제 backend 책임을 분리할 수 있다.

### 5. launcher simplification intuition: `accelerate config/test/launch`가 줄여 주는 것
- Quicktour는 `accelerate config`, `accelerate test`, `accelerate launch`를 핵심 흐름으로 제시한다.
- 이 launch layer의 장점은 실행 환경마다 달라지는 bootstrap 설정을 공통 config와 CLI 흐름으로 다루게 해 준다는 점이다.
- 실무 감각으로 바꾸면 Accelerate는 아래를 단순화한다.
  - single vs multi-GPU vs multi-node vs TPU 환경 차이를 위한 초기 설정
  - launcher 인자/환경 변수 일부
  - backend용 config file 연결 지점
- 하지만 launcher simplification도 한계가 있다.
  - 통신 포트, multi-node 주소, scheduler(SLURM 등) 환경에서는 여전히 시스템 설정 이해가 필요하다.
  - backend-specific config(예: DeepSpeed ZeRO stage, FSDP wrap policy)는 여전히 직접 읽고 조정해야 한다.
  - `accelerate launch`가 돌아간다고 해서 memory fit, communication efficiency, checkpoint compatibility가 자동 해결되는 것은 아니다.
- 따라서 launcher는 **분산 실행 진입장벽을 낮추는 층** 이지, 분산 시스템 자체를 설명해 주는 층은 아니다.

### 6. mixed precision simplification: 쉽게 켤 수 있지만 이해까지 대체하지는 않는다
- `Accelerator(mixed_precision="fp16" | "bf16" | "fp8")`처럼 precision 모드를 한 곳에서 선언할 수 있다는 점은 Accelerate의 큰 장점이다.
- 문서상 `autocast()`와 `backward()`는 이 precision 설정과 연동되어 적절한 casting/scaling 경로를 사용하게 만든다.
- 이로 인해 학습 루프에서 직접 AMP scaffolding을 많이 쓰지 않고도 mixed precision을 실험할 수 있다.
- 하지만 이 단순화가 감추는 현실도 있다.
  - 어떤 hardware에서 fp16/bf16/fp8이 지원되는지
  - overflow가 났는지 (`optimizer_step_was_skipped`) 같은 관찰 포인트
  - 수치 안정성, gradient clipping, loss scaling, checkpoint dtype
  - backend별 precision handling 차이
- 즉 Accelerate는 mixed precision을 **설정하기 쉽게** 만들지만, mixed precision이 시스템에 남기는 흔적을 **이해하지 않아도 되게** 만들지는 않는다.

### 7. Accelerate가 특히 도움이 되는 지점
- single-device training loop를 큰 구조 변경 없이 multi-device run으로 옮기고 싶을 때
- 직접 DDP boilerplate, dataloader sharding, backward 경로 분기를 다 쓰고 싶지 않을 때
- 같은 코드베이스에서 gradient accumulation, mixed precision, tracker, save/load state를 공통 인터페이스로 다루고 싶을 때
- DeepSpeed/FSDP 같은 backend를 직접 통합하기 전, **공통 launch/preparation layer** 를 잡고 싶을 때
- 교육 관점에서는 "분산 학습 환경 차이 때문에 코드가 어떻게 찢어지는가"를 보기보다, 먼저 **학습 루프의 핵심 구조를 유지한 채 어디까지 일반화할 수 있는가** 를 보여 주기 좋다.

### 8. Accelerate가 복잡도를 숨기지만 없애지 않는 지점
- memory ceiling: model/optimizer/activation이 실제로 fit하는지 여부
- communication cost: all-reduce, shard gather, gradient synchronization 비용
- backend semantics: DDP, ZeRO, FSDP가 무엇을 어떻게 분산하는지
- checkpoint semantics: unwrap/save/load/state restore가 어떤 객체 기준으로 이뤄지는지
- evaluation gather: process별 prediction을 metric 계산용으로 다시 모으는 절차
- debugging: deadlock, uneven batch, skipped optimizer step, mixed precision instability
- 즉 Accelerate는 **코드 표면의 복잡도는 줄여 주지만**, 분산 시스템의 물리적·수치적 현실은 그대로 남긴다.

### 9. common confusion
- Accelerate를 쓰면 분산 학습 원리를 몰라도 된다고 생각하는 실수
  - 실제로는 launcher와 prepare를 쉽게 만들 뿐, backend 원리와 병목은 여전히 중요하다.
- `prepare(...)`를 단순 device move helper로 생각하는 실수
  - model/optimizer/scheduler/dataloader의 실행 계약을 다시 감싸는 단계다.
- automatic device placement가 있으니 tensor/device 오류가 완전히 사라진다고 생각하는 실수
  - 수동 배치가 섞이거나 optimizer 생성 순서가 어긋나면 여전히 문제가 생길 수 있다.
- mixed precision을 켰으니 자동으로 더 빠르고 더 안정적일 것이라 생각하는 실수
  - hardware, overflow, kernel support, backend 정책 차이를 계속 봐야 한다.
- `accelerate launch`로 실행되면 backend 차이까지 추상화됐다고 생각하는 실수
  - 실제 메모리 절약 메커니즘과 통신 패턴은 ZeRO/FSDP/DDP 각각 다르다.
- wrapper가 생겼으니 saving/loading/debugging도 완전히 동일하다고 생각하는 실수
  - unwrap, state save/load, gather_for_metrics 같은 후처리 감각이 여전히 필요하다.

## 이 단위에서 무엇을 관찰할 것인가
- 기존 PyTorch loop에서 Accelerate 도입 후 사라지는 코드와 남는 코드는 각각 무엇인가?
- `accelerator.state`, `distributed_type`, `num_processes`, `sync_gradients`는 어떤 실행 상태를 드러내는가?
- `prepare(...)` 전후 dataloader/batch/model wrapper의 관찰 가능한 차이는 무엇인가?
- mixed precision을 켰을 때 단순해지는 API 표면 뒤에서, 어떤 overflow/precision/debugging 질문이 새로 생기는가?
- `accelerate launch`는 launcher complexity를 얼마나 줄이지만, 어떤 system-level 설정은 여전히 이해해야 하는가?
- 다음 backend 단위(DeepSpeed/FSDP)로 넘어갈 때, Accelerate 책임과 backend 책임의 경계를 어떻게 설명할 수 있는가?
