# 03 DeepSpeed ZeRO

> Status: runnable

## 왜 이 단위를 배우는가
큰 모델 학습에서 메모리를 많이 차지하는 것은 parameter만이 아니다. gradient, optimizer state, activation, temporary buffer가 함께 쌓인다. ZeRO는 이 중 일부를 rank별로 나눠 갖게 만들어 한 device가 모든 상태를 들고 있지 않게 한다. 이 단위는 실제 DeepSpeed 없이도 **무엇을 쪼개면 얼마나 줄어드는지**를 deterministic memory accounting으로 확인한다.

## 이번 단위에서 남길 것
- scratch memory accounting `artifacts/scratch-manual/metrics.json`
- ZeRO stage SVG `artifacts/scratch-manual/zero_memory_stages.svg`
- framework-style simulation `artifacts/framework-manual/metrics.json`
- observed report `artifacts/analysis-manual/latest_report.md`
- stable `analysis.md`
- learner worksheet `reflection.md`

## 실습 흐름
1. parameter / gradient / optimizer state 메모리를 따로 계산한다.
2. ZeRO stage 1/2/3이 각각 어떤 상태를 나눠 갖는지 비교한다.
3. communication overhead와 memory saving을 동시에 본다.
4. `analysis.py`로 “메모리 절약”과 “복잡도 증가”를 함께 해석한다.

## 이 단위에서 특히 볼 질문
- optimizer state가 parameter보다 클 수 있는 이유는 무엇인가?
- ZeRO stage가 올라가면 무엇이 더 많이 shard되는가?
- 메모리가 줄어드는 대신 어떤 통신/복구 복잡도가 생기는가?

## 실행 결과 예시
```text
$ python 06_training_systems/03_deepspeed_zero/scratch_lab.py
{
  "dp_baseline_mb": 96.0,
  "zero_stage_1_mb": 48.0,
  "zero_stage_2_mb": 36.0,
  "zero_stage_3_mb": 24.0
}

$ python 06_training_systems/03_deepspeed_zero/framework_lab.py
{
  "backend": "zero-simulated",
  "best_memory_stage": "zero3",
  "communication_penalty_rank": [1, 2, 3]
}
```

## 다음 단위와의 연결
ZeRO가 상태를 shard하는 감각을 잡으면, `04_fsdp_checkpointing_and_offload`에서 FSDP와 checkpoint/offload를 더 구체적으로 비교할 수 있다.
