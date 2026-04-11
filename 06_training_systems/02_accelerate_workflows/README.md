# 02 Accelerate Workflows

> Status: runnable

## 왜 이 단위를 배우는가
`torchrun`과 DDP의 rank 계약을 이해해도, 매번 device placement, mixed precision, launcher 설정을 직접 만지면 학습 루프가 쉽게 복잡해진다. Accelerate는 이 복잡도를 모두 없애는 도구가 아니라, **같은 PyTorch loop를 여러 실행 환경으로 옮길 때 반복되는 보일러플레이트를 줄이는 적응 계층**이다. 이 단위는 실제 Accelerate 설치 없이도 그 추상화 경계를 CPU-safe simulation으로 확인하게 한다.

## 이번 단위에서 남길 것
- scratch simulation metrics `artifacts/scratch-manual/metrics.json`
- workflow SVG `artifacts/scratch-manual/accelerate_workflow.svg`
- framework-style simulation metrics `artifacts/framework-manual/metrics.json`
- observed report `artifacts/analysis-manual/latest_report.md`
- stable `analysis.md`
- learner worksheet `reflection.md`

## 실습 흐름
1. baseline loop에서 직접 device call과 manual backward가 어디 있는지 센다.
2. Accelerate-style loop에서 `prepare`, `backward`, `device placement`, `mixed_precision` 설정이 무엇을 감추는지 비교한다.
3. `framework_lab.py`에서 prepared model/optimizer/dataloader wrapper를 숫자 상태로 다시 표현한다.
4. `analysis.py`로 “숨겨지는 것”과 “여전히 알아야 하는 것”을 나눈다.

## 이 단위에서 특히 볼 질문
- Accelerate는 training loop를 대체하는가, 아니면 실행 환경 적응 계층인가?
- `prepare()`가 감추는 것과 감추지 않는 것은 무엇인가?
- mixed precision 설정을 쉽게 켜도 수치 안정성 이해가 왜 필요한가?

## 실행 결과 예시
```text
$ python 06_training_systems/02_accelerate_workflows/scratch_lab.py
{
  "baseline_explicit_device_calls": 3,
  "accelerate_replaced_calls": 3,
  "distributed_type": "MULTI_GPU",
  "num_processes": 4,
  "mixed_precision": "bf16"
}

$ python 06_training_systems/02_accelerate_workflows/framework_lab.py
{
  "backend": "accelerate-simulated",
  "prepared_object_count": 4,
  "manual_rank_logic_removed": "partially"
}
```

## 다음 단위와의 연결
Accelerate가 launcher와 device boilerplate를 줄여도, ZeRO/FSDP가 실제 메모리를 어떻게 나누는지는 별도 문제다. 그래서 다음 단위 `03_deepspeed_zero`로 넘어간다.
