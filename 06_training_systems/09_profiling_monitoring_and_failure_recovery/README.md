# 09 Profiling, Monitoring, and Failure Recovery

> Status: runnable
>
> 이 단위는 실제 GPU profiler나 distributed runtime 없이 실행되는 **CPU-safe deterministic simulation**이다. 목적은 도구 하나를 외우는 것이 아니라, training step의 time / memory / communication 신호를 한 장의 runbook으로 묶고 OOM, hang, divergence, checkpoint failure를 서로 다른 복구 결정으로 연결하는 것이다.

## 왜 이 단위를 배우는가
`06_training_systems` 앞 단위들에서 우리는 DDP, ZeRO, FSDP, tensor parallel, pipeline parallel 같은 **구성 방법**을 배웠다. 하지만 실제 large-model 학습에서는 구성을 아는 것만으로 충분하지 않다. 학습이 느려지거나 멈추거나 품질이 무너지기 시작하면, 이제 질문은 "어떤 기법을 켰는가"가 아니라 **시간은 어디서 쓰였는가, 메모리는 언제 치솟았는가, 어떤 rank가 먼저 이상 신호를 냈는가, 어디서부터 다시 시작해야 하는가**로 바뀐다.

profiling, monitoring, failure recovery는 서로 떨어진 주제가 아니다. profiler는 병목의 모양을 보여 주고, monitoring은 그 병목이 반복되는지 알려 주며, recovery는 문제가 터진 뒤 실험을 어떻게 안전하게 이어 갈지 결정한다. 이 단위는 training systems 트랙의 마지막에서 앞선 병렬화/메모리 기법을 **관찰 가능하고 복구 가능한 학습 시스템**으로 묶는다.

## 이번 단위에서 남길 것
- `scratch_lab.py` — 8-step training window를 deterministic하게 profile하고 throughput, step time p50/p95, time breakdown, memory snapshot, rank heartbeat, alert를 `artifacts/scratch-manual/metrics.json`에 쓴다.
- `artifacts/scratch-manual/profiling_timeline.svg` — step time과 communication wait가 커지는 구간을 눈으로 확인하는 작은 SVG timeline이다.
- `framework_lab.py` — monitoring contract, failure taxonomy, checkpoint manifest, retry policy, recovery decision을 framework-style 운영 runbook으로 시뮬레이션한다.
- `analysis.py` — 두 metrics 파일을 읽어 `artifacts/analysis-manual/latest_report.md`에 관측 보고서를 쓴다. metrics가 없으면 먼저 실행할 명령을 알려 주며 실패한다.
- `analysis.md` — 실행해도 바뀌지 않는 stable interpretation 문서다.
- `reflection.md` — 실행 전 예측과 실행 후 triage/recovery 회고 질문이다.

## 실행 방법
아래 명령은 모두 저장소 루트에서 실행한다.

```bash
python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py
python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/framework_lab.py
python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/analysis.py
```

생성되는 산출물은 다음 위치에 고정된다.

```text
06_training_systems/09_profiling_monitoring_and_failure_recovery/artifacts/
├── scratch-manual/
│   ├── metrics.json
│   └── profiling_timeline.svg
├── framework-manual/
│   └── metrics.json
└── analysis-manual/
    └── latest_report.md
```

## 실행 결과 예시

```text
$ python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/scratch_lab.py
{
  "status": "runnable",
  "cpu_safe_simulation": true,
  "profile_window": {
    "steps": 8,
    "world_size": 4,
    "tokens_per_step": 4096
  },
  "dominant_bottleneck": "communication_wait_due_to_rank_2_heartbeat_lag",
  "alerts": [
    "throughput_drop_gt_20pct",
    "step_time_p95_exceeds_p50_by_30pct",
    "rank_2_heartbeat_lag",
    "checkpoint_age_exceeds_target"
  ]
}

$ python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/framework_lab.py
{
  "status": "runnable",
  "framework": "cpu_deterministic_monitoring_recovery_sim",
  "failure_triage": {
    "selected_incident": {
      "classification": "hang_or_straggler"
    }
  },
  "recovery_decision": {
    "action": "retry_from_last_good_checkpoint",
    "post_resume_validation": {
      "passed": true
    }
  }
}

$ python3 06_training_systems/09_profiling_monitoring_and_failure_recovery/analysis.py
# 09 Profiling, Monitoring, and Failure Recovery 실행 관측
...
```

## 실습 흐름
1. `scratch_lab.py`의 step timeline을 보며 평균 step time이 아니라 p95, second-half jitter, communication wait 증가를 먼저 확인한다.
2. throughput drop과 rank heartbeat lag가 같이 움직이는지 보고, compute hotspot인지 communication wait / straggler인지 첫 가설을 세운다.
3. memory snapshot에서 allocated, reserved, peak phase를 분리해 OOM 위험이 batch 자체인지 checkpoint/eval boundary인지 해석한다.
4. `framework_lab.py`의 monitoring contract를 읽고 throughput, loss/grad norm, memory, heartbeat, checkpoint freshness가 한 snapshot에 왜 같이 있어야 하는지 확인한다.
5. failure taxonomy에서 OOM, hang, divergence, storage/checkpoint failure의 첫 질문이 어떻게 다른지 비교한다.
6. checkpoint manifest와 retry policy를 보고, retry가 안전한 경우와 checkpoint quarantine이 필요한 경우를 나눈다.
7. `analysis.py` 보고서를 읽고 병목 진단 → failure triage → recovery decision이 한 runbook으로 이어지는지 점검한다.

## 이 단위에서 특히 볼 질문
- 학습이 느리다는 말은 정확히 어떤 시간축 문제인가? data wait, compute, communication wait, checkpoint I/O를 어떻게 구분할 수 있는가?
- 평균 step time이 비슷해 보여도 throughput이 흔들리거나 tail latency가 커질 때, 무엇을 먼저 의심해야 하는가?
- `allocated` 와 `reserved` 메모리 차이, peak 시점, fragmentation 징후를 함께 보면 어떤 OOM 위험을 더 빨리 읽을 수 있는가?
- hang와 "매우 느린 step"은 어떻게 구분해야 하며, 어떤 rank-level signal이 deadlock/timeout/straggler를 구분하게 도와주는가?
- loss spike, NaN, grad norm explosion, resume 직후 품질 붕괴는 왜 모두 같은 divergence로 묶을 수 없고, 어떤 순서로 원인을 좁혀야 하는가?
- checkpoint가 저장되었다는 사실과 실제로 재시작 가능한 checkpoint라는 사실은 왜 다른가?
- retry/recovery decision은 어떤 관측과 post-resume validation이 있어야 자동화할 수 있는가?

## 다음 단위와의 연결
이 트랙의 마지막 단위로서, 여기서 profiling·monitoring·failure recovery 감각을 잡아 두면 다음 학습 단계인 `07_frontier_labs/01_paper_reproduction_playground`에서 훨씬 현실적인 실험 운영이 가능해진다. 논문 재현이나 capstone build는 아이디어만으로 굴러가지 않고, 느려진 실험을 해석하고, 실패를 분류하고, 마지막 good checkpoint에서 안전하게 다시 시작하는 운영 습관이 있어야 지속된다.
