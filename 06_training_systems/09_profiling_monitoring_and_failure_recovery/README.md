# 09 Profiling, Monitoring, and Failure Recovery

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조**이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
`06_training_systems` 앞 단위들에서 우리는 DDP, ZeRO, FSDP, tensor parallel, pipeline parallel 같은 **구성 방법**을 배웠다. 하지만 실제 large-model 학습에서는 구성을 아는 것만으로 충분하지 않다. 학습이 느려지거나 멈추거나 품질이 무너지기 시작하면, 이제 질문은 "어떤 기법을 켰는가"가 아니라 **시간은 어디서 쓰였는가, 메모리는 언제 치솟았는가, 어떤 rank가 먼저 이상 신호를 냈는가, 어디서부터 다시 시작해야 하는가**로 바뀐다. 이 단위는 바로 그 운영 관점을 정리한다.

또한 profiling, monitoring, failure recovery는 서로 떨어진 주제가 아니다. profiler는 병목의 모양을 보여 주고, monitoring은 그 병목이 언제 반복적으로 나타나는지 알려 주며, recovery는 문제가 터진 뒤 실험을 어떻게 안전하게 이어 갈지 결정한다. 즉 이 단위는 training systems 트랙의 마지막에서, 앞선 병렬화/메모리 기법들을 **관찰·진단·복구 가능한 학습 시스템**으로 묶는 역할을 한다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- time / memory / communication profiling intuition과 운영 해석을 정리한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 실습 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - per-step time / memory / communication breakdown 요약
  - throughput, loss, grad norm, GPU util, checkpoint freshness를 묶은 monitoring snapshot
  - OOM / hang / divergence incident triage 메모
  - checkpoint resume 검증 체크리스트와 recovery decision log

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 training step 하나를 `data wait → forward/backward compute → communication → checkpoint/save I/O` 같은 시간축으로 나눠 보고, profiling이 왜 단순 "느린 함수 찾기"보다 **step lifecycle 해석**에 가까운지 정리한다.
2. profiler trace, memory snapshot, throughput log를 함께 놓고, 같은 slowdown이라도 dataloader stall인지, GPU kernel 병목인지, collective wait인지, checkpoint flush 때문인지 구분하는 기준선을 만든다.
3. monitoring 관점에서 loss, grad norm, step time jitter, GPU utilization, peak reserved memory, rank heartbeat, last-good-checkpoint 시각 중 무엇을 계속 봐야 하는지 묶어 본다.
4. failure taxonomy를 OOM, hang, divergence, storage/checkpoint failure, preemption/restart 문제로 나누고, 각 경우에 가장 먼저 확인할 관찰 포인트가 무엇인지 정리한다.
5. checkpoint/restart 관점에서 모델 파라미터만이 아니라 optimizer/scheduler/scaler/data position/RNG/state manifest까지 이어지는 **recoverable run contract**를 점검한다.
6. 마지막에는 이 단위에서 만든 profiling·monitoring·recovery runbook이 다음 트랙 `07_frontier_labs`의 paper reproduction, capstone build, agentic eval loop 운영에서 왜 필수인지 연결한다.

## 이 단위에서 특히 볼 질문
- 학습이 느리다는 말은 정확히 어떤 시간축 문제인가? data wait, compute, communication, checkpoint I/O를 어떻게 구분할 수 있는가?
- 평균 step time이 비슷해 보여도 throughput이 흔들리거나 tail latency가 커질 때, 무엇을 먼저 의심해야 하는가?
- `allocated` 와 `reserved` 메모리 차이, peak 시점, fragmentation 징후를 함께 보면 어떤 OOM 위험을 더 빨리 읽을 수 있는가?
- hang와 "매우 느린 step"은 어떻게 구분해야 하며, 어떤 rank-level signal이 deadlock/timeout/straggler를 구분하게 도와주는가?
- loss spike, NaN, grad norm explosion, resume 직후 품질 붕괴는 왜 모두 같은 divergence로 묶을 수 없고, 어떤 순서로 원인을 좁혀야 하는가?
- checkpoint가 저장되었다는 사실과 실제로 재시작 가능한 checkpoint라는 사실은 왜 다른가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py
{
  "status": "sample",
  "profile_window": {
    "steps": 120,
    "world_size": 8,
    "global_batch": 512
  },
  "time_breakdown": {
    "data_wait_pct": 11.8,
    "forward_backward_compute_pct": 54.2,
    "communication_wait_pct": 22.6,
    "checkpoint_or_misc_io_pct": 11.4
  },
  "memory_snapshot": {
    "peak_allocated_gb": 71.3,
    "peak_reserved_gb": 79.8,
    "peak_step": 438,
    "fragmentation_hint": "reserved remains high after eval+save boundary"
  },
  "alerts": [
    "step_time_jitter_visible_after_step_430",
    "rank_6_collective_wait_spike",
    "checkpoint_age_exceeds_target"
  ],
  "notes": [
    "expected output/sample shape only",
    "profiling result is interpreted together with monitoring signals"
  ]
}

$ python 06_training_systems/09_profiling_monitoring_and_failure_recovery/framework_lab.py
{
  "status": "sample",
  "failure_triage": {
    "symptom": "loss_nan_after_resume",
    "first_split": "divergence_not_hang",
    "immediate_checks": [
      "last_verified_checkpoint_id",
      "optimizer_and_scheduler_step_continuity",
      "grad_norm_spike_before_failure",
      "all_ranks_loaded_same_manifest"
    ]
  },
  "monitoring_contract": {
    "required_signals": [
      "throughput",
      "step_time_p50_p95",
      "loss_and_grad_norm",
      "gpu_memory_allocated_reserved",
      "per-rank heartbeat",
      "checkpoint_freshness"
    ],
    "alert_examples": [
      "throughput_drop_gt_20pct",
      "memory_growth_without_recovery",
      "no_checkpoint_progress_for_30min"
    ]
  },
  "recovery_plan": {
    "resume_from": "last_good_checkpoint",
    "checkpoint_format": "sharded_state_plus_manifest",
    "post_resume_validation": [
      "global_step_matches",
      "sampler_state_restored",
      "lr_scheduler_state_restored",
      "first_10_steps_are_numerically_stable"
    ]
  }
}
```

핵심은 profiler 출력 한 장을 읽는 것이 아니라, **병목이 시간·메모리·통신 중 어디서 나타나는지**, **지속적으로 봐야 할 monitoring 신호가 무엇인지**, **장애 뒤에 어떤 checkpoint에서 어떤 검증을 거쳐 다시 출발해야 하는지**를 한 runbook으로 연결하는 것이다.

## 다음 단위와의 연결
이 트랙의 마지막 단위로서, 여기서 profiling·monitoring·failure recovery 감각을 잡아 두면 다음 학습 단계인 `07_frontier_labs/01_paper_reproduction_playground`에서 훨씬 현실적인 실험 운영이 가능해진다. 논문 재현이나 capstone build는 아이디어만으로 굴러가지 않고, 느려진 실험을 해석하고, 실패를 분류하고, 마지막 good checkpoint에서 안전하게 다시 시작하는 운영 습관이 있어야 지속된다.

즉 이 단위는 `06_training_systems`에서 배운 병렬화/메모리 전략들을 단순 설정 모음으로 끝내지 않고, 이후 frontier-style 실험에서 **관찰 가능성(observability)과 복구 가능성(recoverability)** 을 갖춘 시스템 실천으로 넘겨 주는 연결 고리다.
