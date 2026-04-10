# 06 Pipeline Parallelism

> Status: planned

## 왜 이 단위를 배우는가
이 단위는 모델 레이어를 stage로 나누는 pipeline parallel이 throughput과 latency에 어떤 영향을 주는지 이해하게 한다.

## 이번 단위에서 들어올 것
- stage partitioning
- microbatch scheduling
- pipeline bubble intuition
- inter-stage checkpoint design

## 선행 개념
- `06_training_systems/05_tensor_parallelism`
- `02_deep_learning/04_attention_and_transformers`
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
