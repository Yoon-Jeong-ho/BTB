# 05 Tensor Parallelism

> Status: planned

## 왜 이 단위를 배우는가
이 단위는 레이어 내부 행렬 연산 자체를 여러 장치로 나누는 tensor parallel 구조를 이해하도록 돕는다.

## 이번 단위에서 들어올 것
- row/column parallel linear layers
- intra-layer communication
- attention/feed-forward partitioning
- tensor-parallel topology sketch

## 선행 개념
- `02_deep_learning/04_attention_and_transformers`
- `06_training_systems/04_fsdp_checkpointing_and_offload`
- `00_foundations/01_tensor_shapes`

## 계획된 산출물
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.md`
- `reflection.md`
