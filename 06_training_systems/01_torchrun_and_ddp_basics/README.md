# 01 Torchrun and DDP Basics

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 runnable/applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
단일 GPU 또는 단일 프로세스 학습 스크립트는 보통 "모델 하나, optimizer 하나, Python 프로세스 하나"라는 감각으로 이해된다. 그런데 실제 대규모 학습으로 넘어가면 같은 모델이라도 **여러 프로세스를 어떤 규칙으로 띄우고, 각 프로세스가 어떤 GPU와 데이터를 맡으며, gradient를 언제 어떻게 맞춰야 하는가**가 핵심 계약이 된다. 이 단위는 그 출발점으로서 `torchrun`과 Distributed Data Parallel(DDP)을 블랙박스 도구가 아니라 **분산 학습의 가장 작은 실행 계약**으로 이해하게 만든다.

또한 이후 `06_training_systems`의 Accelerate, ZeRO, FSDP, hybrid parallelism을 보려면 먼저 `world size`, `rank`, `local rank`, main-process logging 같은 기본 용어가 몸에 들어와 있어야 한다. 이 단위는 "GPU를 더 많이 쓰는 법" 자체보다, **여러 프로세스가 동시에 같은 학습을 수행할 때 무엇이 달라지고 무엇은 같아야 하는가**를 읽는 첫 시스템 단위다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- single-process 대비 distributed launch 직관과 DDP 통신 감각을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - per-rank environment snapshot (`rank`, `local_rank`, `world_size`, device)
  - main-rank-only logging / checkpointing 예시
  - DDP gradient synchronization 관찰 메모
  - effective global batch 해석 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 단일 프로세스 학습 루프를 먼저 떠올리며, `python train.py`처럼 실행했을 때는 왜 process가 하나뿐이고 rank 개념이 필요 없는지 정리한다.
2. 같은 학습을 `torchrun --nproc_per_node=...`로 띄운다고 가정하고, 각 worker process에 `WORLD_SIZE`, `RANK`, `LOCAL_RANK`가 어떻게 배정되는지 본다.
3. `local_rank`가 "이 프로세스가 현재 노드에서 어느 장치를 맡는가"를 의미한다는 점을 확인하고, 보통 one-process-per-GPU 패턴이 왜 기본이 되는지 연결한다.
4. 각 rank가 서로 다른 mini-batch shard를 처리한 뒤 backward 시점에 gradient를 맞추면, 왜 optimizer step은 각 프로세스가 따로 해도 파라미터가 같은 방향으로 유지되는지 DDP intuition을 잡는다.
5. 로그, evaluation, checkpoint 저장은 왜 보통 rank 0 또는 main process만 담당하게 되는지 보고, 모든 rank가 같은 일을 중복하면 어떤 혼란이 생기는지도 관찰한다.
6. 마지막에는 이 launch/통신 계약이 이후 `06_training_systems/02_accelerate_workflows`, `06_training_systems/07_data_parallel_grad_accumulation`에서 어떻게 더 추상화되거나 확장되는지 연결한다.

## 이 단위에서 특히 볼 질문
- 단일 프로세스 학습과 distributed launch의 가장 큰 차이는 "GPU 수"가 아니라 어떤 실행 계약의 변화인가?
- `world_size`, `rank`, `local_rank`는 각각 무엇을 세며, 언제 서로 같은 숫자처럼 보이고 언제 달라지는가?
- 왜 `torchrun`이 단순 shell convenience가 아니라 distributed process group의 기본 launcher 역할을 하는가?
- DDP는 각 프로세스가 독립적으로 학습하는 것처럼 보여도 어떻게 파라미터 동기화를 유지하는가?
- 왜 local batch size와 global/effective batch size를 구분해야 하며, 분산 환경에서 loss 해석이 어떻게 달라질 수 있는가?
- 왜 logging, evaluation, checkpointing을 모든 rank에서 동시에 하지 않고 main process 기준으로 정리하는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ torchrun --standalone --nproc_per_node=2 06_training_systems/01_torchrun_and_ddp_basics/scratch_lab.py
[
  {
    "rank": 0,
    "local_rank": 0,
    "world_size": 2,
    "device": "cuda:0",
    "is_main_process": true,
    "local_batch_shape": [8, 128]
  },
  {
    "rank": 1,
    "local_rank": 1,
    "world_size": 2,
    "device": "cuda:1",
    "is_main_process": false,
    "local_batch_shape": [8, 128]
  }
]

$ torchrun --standalone --nproc_per_node=2 06_training_systems/01_torchrun_and_ddp_basics/framework_lab.py
{
  "status": "sample",
  "launcher": "torchrun",
  "ddp_config": {
    "backend": "nccl",
    "world_size": 2,
    "gradient_sync": "allreduce_mean"
  },
  "per_rank_loss": [1.92, 1.95],
  "post_sync_parameter_checksum": "same_across_ranks",
  "global_batch_size": 16,
  "notes": "expected output/sample shape only"
}
```

핵심은 숫자 자체보다도 **각 프로세스가 어떤 identity를 받고 있는지**, **backward 뒤 gradient가 어떤 방식으로 맞춰진다고 이해해야 하는지**, **main-rank-only 관찰 규칙이 왜 필요한지**를 읽는 것이다.

## 다음 단위와의 연결
이 단위는 `06_training_systems` 전체의 공통 바닥이다. 여기서 `torchrun`, `rank`, `local_rank`, DDP gradient sync 감각을 잡아 두면 다음 단위들이 훨씬 덜 추상적으로 보인다.

- `06_training_systems/02_accelerate_workflows`에서는 이 launch/장치 배치를 더 높은 추상화로 감싼다는 관점으로 이어진다.
- `06_training_systems/07_data_parallel_grad_accumulation`에서는 local batch와 global/effective batch 구분이 실제 optimizer step 해석으로 확장된다.
- `06_training_systems/09_profiling_monitoring_and_failure_recovery`에서는 main-rank logging, failure triage, checkpoint 복구가 운영 runbook 형태로 구체화된다.

즉 이 단위는 "분산 학습이 어렵다"는 막연함을 줄이고, **여러 프로세스가 같은 모델을 함께 학습시킬 때 꼭 고정해야 하는 최소 좌표계**를 마련하는 첫 단계다.
