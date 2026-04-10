# 03 DeepSpeed ZeRO

> Status: planned

## 왜 이 단위를 배우는가
이 단위는 optimizer/model/gradient state를 어떻게 shard해서 메모리 병목을 줄이는지 이해하도록 돕는다.

## 이번 단위에서 들어올 것
- ZeRO stage 1/2/3 intuition
- state sharding
- memory vs communication trade-off
- configuration sketch

## 선행 개념
- `06_training_systems/01_torchrun_and_ddp_basics`
- `06_training_systems/02_accelerate_workflows`
- `00_foundations/05_gpu_memory_runtime`

## 계획된 산출물
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.md`
- `reflection.md`
