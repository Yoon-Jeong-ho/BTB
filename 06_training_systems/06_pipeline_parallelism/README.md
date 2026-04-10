# 06 Pipeline Parallelism

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
모델이 커질수록 "한 장치가 레이어 전체를 다 들고 forward/backward를 끝낸다"는 가정이 깨지기 쉽다. 이때 pipeline parallelism은 레이어 스택을 여러 **stage**로 나누고, microbatch를 stage 사이로 흘려 보내면서 **메모리 적재와 처리 순서 자체를 다시 설계하는 방법**으로 등장한다. 즉 이 단위는 분산 학습을 단순히 데이터 복제나 state sharding으로만 보지 않고, **모델 실행 경로를 시간축까지 포함해 분할하는 사고**로 확장하게 만든다.

또한 later LLM/대형 transformer 학습에서는 tensor parallel, data parallel, FSDP, ZeRO가 서로 섞여 등장한다. pipeline parallel을 따로 이해해 두면 "레이어를 stage로 나누는 것"과 "레이어 내부 텐서를 나누는 것", "모델 복제본을 여러 rank에 두는 것"을 구분해 볼 수 있고, 이후 `06_training_systems/08_hybrid_parallel_topologies`에서 왜 여러 병렬화 축을 조합하는지 훨씬 명확해진다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- pipeline stage intuition, bubble/throughput, microbatch scheduling basics를 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - stage partition 스케치와 stage별 layer/cost 메모
  - warmup / steady-state / cooldown schedule trace
  - bubble fraction 및 throughput 관찰 요약
  - activation send/recv 경계와 partition boundary 주의점 메모

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 transformer block이나 MLP stack처럼 순차적인 레이어 흐름을 한 줄로 놓고, 어느 지점에서 stage를 자르면 각 장치가 어떤 연산 구간을 맡게 되는지 그려 본다.
2. single batch를 한 번 흘리는 경우와 microbatch 여러 개를 쪼개서 흘리는 경우를 비교하며, pipeline parallel이 왜 **memory fit**과 **device utilization** 관점에서 의미가 생기는지 본다.
3. warmup, steady state, cooldown을 시간축으로 적어 보며 pipeline bubble이 언제 생기고, microbatch 수가 늘면 왜 idle 구간 비율이 줄어드는지 계산한다.
4. GPipe식 all-forward-then-all-backward와 1F1B(one-forward-one-backward) 같은 기본 스케줄을 비교하며, activation 메모리와 throughput trade-off가 어떻게 달라지는지 정리한다.
5. stage 경계에서 activation tensor를 send/recv한다고 가정하고, boundary mismatch·skip connection·불균형 partition이 어떤 문제를 만드는지 본다.
6. 마지막에는 이 감각을 바탕으로 다음 단위 `06_training_systems/07_data_parallel_grad_accumulation`, `08_hybrid_parallel_topologies`와 연결해, pipeline이 단독 해법이 아니라 다른 병렬화 축과 결합되는 이유를 정리한다.

## 이 단위에서 특히 볼 질문
- pipeline parallelism은 "모델을 여러 GPU에 나눈다"는 말보다 더 구체적으로 어떤 실행 계약 변화를 뜻하는가?
- stage partition이 메모리 적재량을 줄이면서도, 왜 single-batch latency를 자동으로 줄여 주지는 않는가?
- pipeline bubble은 정확히 어느 구간의 idle time을 뜻하며, microbatch 수가 늘면 왜 bubble fraction이 줄어드는가?
- GPipe식 스케줄과 1F1B 스케줄은 activation 보관량과 throughput 감각에서 어떻게 다른가?
- stage 경계에서 activation transfer가 생기면, 어떤 shape/dtype/communication 질문이 새로 중요해지는가?
- layer 수를 균등하게 나누는 것과 stage 시간을 균등하게 맞추는 것은 왜 다른 문제인가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 06_training_systems/06_pipeline_parallelism/scratch_lab.py
{
  "status": "sample",
  "model": {
    "num_layers": 24,
    "num_stages": 4,
    "layers_per_stage": [6, 6, 6, 6],
    "microbatches": 8
  },
  "schedule_summary": {
    "policy": "1F1B",
    "warmup_steps": 3,
    "steady_state_steps": 8,
    "cooldown_steps": 3,
    "estimated_bubble_fraction": 0.27
  },
  "stage_observations": [
    {"stage": 0, "dominant_work": "embedding + early blocks", "idle_slots": 3},
    {"stage": 1, "dominant_work": "middle blocks", "idle_slots": 1},
    {"stage": 2, "dominant_work": "late blocks", "idle_slots": 1},
    {"stage": 3, "dominant_work": "head + loss", "idle_slots": 3}
  ],
  "notes": [
    "sample numbers for intuition only",
    "more microbatches reduce bubble ratio but raise scheduling overhead"
  ]
}

$ python 06_training_systems/06_pipeline_parallelism/framework_lab.py
{
  "status": "sample",
  "partition_plan": {
    "boundary_layers": [6, 12, 18],
    "activation_transfer": "send/recv between adjacent stages",
    "checkpoint_policy": "stage-local + global metadata"
  },
  "runtime_observations": {
    "throughput_gain": "visible after pipeline fill",
    "single_batch_latency": "not necessarily improved",
    "activation_memory_pressure": "depends on schedule and in-flight microbatches",
    "load_balance_risk": "high when stage compute times differ"
  },
  "notes": "expected output/sample shape only"
}
```

핵심은 숫자 하나보다도 **어디서 stage를 자를지**, **microbatch를 어떻게 흘릴지**, **idle bubble과 activation transfer가 어떤 시스템 흔적을 남기는지**를 읽는 것이다.

## 다음 단위와의 연결
이 단위 다음에는 `06_training_systems/07_data_parallel_grad_accumulation`을 통해 data-parallel 축에서 effective batch와 optimizer step cadence를 다시 정리하게 된다. pipeline parallel이 **모델 실행 경로를 stage로 나누는 방법**이라면, 다음 단위는 **같은 모델 복제본을 여러 rank에 두고 step 타이밍을 조정하는 방법**을 더 또렷하게 잡아 준다.

그리고 이후 `06_training_systems/08_hybrid_parallel_topologies`에서는 tensor parallel, pipeline parallel, data parallel을 왜 한꺼번에 조합해야 하는지 본격적으로 연결된다. 즉 이 단위는 "모델을 stage로 나눠 시간축에서 흘린다"는 감각을 만들고, 다음 단위들은 그 위에 **batch 축, layer 내부 축, state sharding 축을 어떻게 겹쳐 설계하는가**로 넘어가는 다리 역할을 한다.
