# 01 Torchrun and DDP Basics 이론 노트

## 핵심 개념

### 1. single-process 학습과 distributed launch는 무엇이 다른가
- single-process 학습에서는 보통 Python interpreter 하나가 모델, optimizer, dataloader loop를 모두 들고 있다.
- 이때는 "누가 몇 번째 worker인가"를 따로 식별할 필요가 없다. process가 하나뿐이기 때문이다.
- distributed launch에서는 같은 프로그램을 **여러 프로세스로 동시에 띄우고**, 각 프로세스가 서로 다른 장치/데이터 shard를 맡는다.
- 특히 DDP basics에서는 보통 **one process per GPU** 패턴을 쓴다. 즉 GPU가 4개면 Python process도 4개가 생긴다.
- 중요한 차이는 "GPU 수가 늘었다"보다 **실행의 주체가 하나에서 여러 개로 바뀌었다**는 점이다. 그래서 process identity, rendezvous, synchronization 같은 개념이 필요해진다.

### 2. distributed launch intuition: 같은 코드가 여러 worker로 복제된다
- `torchrun --nproc_per_node=4 train.py` 같은 실행은 `train.py`를 네 번 독립적으로 띄우는 감각에 가깝다.
- 각 프로세스는 자기 메모리 공간, 자기 optimizer 객체, 자기 dataloader iterator를 가진다.
- 하지만 DDP에서는 모델 파라미터를 동일하게 시작하고 backward 때 gradient를 동기화하므로, 결과적으로 각 프로세스의 optimizer step이 **거의 같은 파라미터 업데이트**를 수행하게 된다.
- 그래서 분산 학습은 "하나의 거대한 process"라기보다, **여러 개의 비슷한 process가 통신 계약을 통해 같은 학습을 맞춰 가는 구조**로 이해하는 편이 낫다.

### 3. world size / rank / local rank basics
- `world_size`
  - 전체 worker process 수다.
  - 단일 노드 4 GPU에서 one-process-per-GPU면 보통 `world_size = 4`다.
  - multi-node에서는 모든 노드의 프로세스를 합친 총합이다.
- `rank`
  - 전체 world 안에서 각 프로세스에 주어지는 **전역 번호**다.
  - 보통 `0`부터 `world_size - 1`까지 매겨진다.
  - 관례적으로 rank 0을 main process처럼 취급해 logging/checkpoint/eval 결과 집계를 맡기는 경우가 많다.
- `local_rank`
  - 현재 노드 안에서의 **로컬 번호**다.
  - 단일 노드에서는 rank와 local_rank가 비슷하게 보여 헷갈리지만, multi-node로 가면 다를 수 있다.
  - 보통 `local_rank`를 이용해 `cuda:{local_rank}` 장치에 process를 붙인다.
- 자주 쓰는 직관
  - `world_size`: 몇 명이 같이 일하는가
  - `rank`: 그중 내가 전체에서 몇 번인가
  - `local_rank`: 현재 서버 안에서 내가 몇 번째 장치를 맡는가

### 4. 왜 torchrun이 필요한가
- 이론적으로는 직접 process를 띄우고 environment variable을 손으로 넣어도 분산 학습을 구성할 수 있다.
- 하지만 실제로는 각 프로세스가 다음 정보를 안정적으로 공유해야 한다.
  - 내가 몇 번째 rank인가
  - 전체 world size는 얼마인가
  - 어느 주소/포트에서 rendezvous할 것인가
  - 현재 노드의 local rank는 무엇인가
- `torchrun`은 이 초기화 정보를 표준 방식으로 worker들에게 전달해 주는 launcher다.
- 그래서 `torchrun`의 핵심 역할은 단순히 "여러 Python 프로세스를 띄운다"를 넘어서, **distributed process group이 서로를 찾고 같은 world에 속한다고 합의하게 만드는 것**이다.
- 이후 elastic/restart 지원 같은 확장 기능도 있지만, 이 단위의 핵심은 먼저 `torchrun`을 **DDP용 표준 진입점**으로 이해하는 것이다.

### 5. DDP high-level communication intuition
- DDP에서 각 rank는 보통 같은 모델 복제본을 가진다.
- forward는 각 rank가 **자기 local mini-batch**로 따로 수행한다.
- backward가 시작되면 각 rank는 자기 데이터에 대한 gradient를 계산한다.
- 이때 DDP는 rank들 사이에서 gradient를 통신해 맞춘다. 직관적으로는 **모든 rank의 gradient를 모아 평균 내고, 그 결과를 각 rank가 같은 값으로 받는 과정**으로 이해하면 된다.
- 그러면 optimizer step은 각 프로세스가 따로 수행해도 입력 gradient가 같으므로 파라미터도 같은 방향으로 업데이트된다.
- 즉 DDP의 핵심은 "모델 파라미터를 매 step마다 통째로 복사한다"보다, **backward 과정에서 gradient 계약을 동기화해 각 rank의 optimizer step을 사실상 동일하게 만드는 것**이다.

### 6. local batch, global batch, data shard를 함께 봐야 한다
- single-process에서는 batch size 하나만 생각해도 되는 경우가 많다.
- DDP에서는 각 rank가 따로 batch를 들고 오므로, 관찰해야 할 batch 개념이 늘어난다.
  - local batch size: 각 rank가 한 번에 처리하는 샘플 수
  - global/effective batch size: local batch × world size (필요하면 grad accumulation까지 포함)
- 예를 들어 rank마다 batch 8을 처리하고 world_size가 4면, 한 optimizer step이 반영하는 샘플 규모는 사실상 32가 된다.
- 그래서 분산 학습 로그를 볼 때는 single-process와 숫자를 직접 비교하기보다, **local batch와 effective batch가 어떻게 바뀌었는지** 먼저 확인해야 한다.
- 또한 data parallel에서는 각 rank가 서로 다른 데이터 shard를 보게 만들어야 하므로, later unit에서는 `DistributedSampler` 같은 개념도 자연스럽게 필요해진다.

### 7. main process 관찰 규칙이 왜 필요한가
- rank가 여러 개면 모든 rank가 같은 로그를 동시에 찍어 output이 섞일 수 있다.
- checkpoint를 모든 rank가 동시에 저장하면 같은 파일을 덮어쓰거나 불필요한 중복 파일이 쌓일 수 있다.
- evaluation 결과도 모든 rank가 따로 출력하면 어떤 숫자가 최종 값인지 헷갈릴 수 있다.
- 그래서 보통은 rank 0 또는 main process에만 다음 책임을 준다.
  - 대표 로그 출력
  - checkpoint 저장
  - progress bar 표시
  - 요약 metric 기록
- 단, 디버깅 단계에서는 per-rank print가 필요할 수 있으므로, "평소에는 rank 0만 요약하고, 문제를 볼 때만 제한적으로 rank별 관찰"이라는 운영 감각이 중요하다.

### 8. 자주 헷갈리는 지점
- `world_size`를 "GPU 수"와 완전히 같은 말로 생각하는 실수
  - 자주 일치하지만, 본질은 **전체 process 수**다.
- `rank`와 `local_rank`를 항상 같은 값으로 생각하는 실수
  - 단일 노드에서는 비슷해 보여도 multi-node에서는 분리된다.
- DDP를 model parallel로 오해하는 실수
  - DDP는 보통 **모델 복제 + 데이터 분할**에 가깝다. 모델을 층별로 나누는 것은 다른 문제다.
- DDP가 매 step마다 파라미터 전체를 수동으로 복사한다고 이해하는 실수
  - 핵심 관찰 포인트는 gradient synchronization 쪽이다.
- rank가 여러 개면 학습이 "자동으로 더 좋아진다"고 생각하는 실수
  - throughput과 batch regime이 바뀌는 것이지, optimization 자체가 공짜로 해결되는 것은 아니다.
- 모든 rank가 같은 데이터를 봐도 괜찮다고 생각하는 실수
  - 그러면 data parallel의 이점이 줄고 로그 해석도 왜곡된다.

## 무엇을 관찰할 것인가
- `torchrun`으로 띄웠을 때 각 프로세스가 어떤 `rank/local_rank/world_size`를 받는가?
- 각 rank가 다른 장치를 잡고 있는가, 아니면 잘못 같은 장치를 공유하고 있는가?
- local batch와 effective global batch를 구분해 설명할 수 있는가?
- backward 뒤 각 rank의 parameter checksum이나 gradient norm이 같은 방향으로 맞춰지는가?
- main process만 로그를 찍게 했을 때와 모든 rank가 동시에 찍을 때 관찰 가능성이 어떻게 달라지는가?
- 이후 Accelerate, ZeRO, FSDP를 볼 때도 결국 같은 launch/identity/sync 질문으로 환원된다는 점이 보이는가?
