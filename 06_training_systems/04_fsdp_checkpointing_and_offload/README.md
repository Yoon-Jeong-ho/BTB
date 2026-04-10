# 04 FSDP, Checkpointing, and Offload

> Status: planned

## 왜 이 단위를 배우는가
이 단위는 FSDP와 activation checkpoint/offload를 묶어 큰 모델을 한정된 메모리에서 다루는 실전 감각을 만든다.

## 이번 단위에서 들어올 것
- parameter sharding with FSDP
- activation checkpointing
- CPU / NVMe offload intuition
- checkpoint save/load contract

## 선행 개념
- `06_training_systems/01_torchrun_and_ddp_basics`
- `06_training_systems/03_deepspeed_zero`
- `02_deep_learning/07_training_recipes_and_debugging`

## 계획된 산출물
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.md`
- `reflection.md`
