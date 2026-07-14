# 04 GPU / Conda Experiment Plan

## 현재 환경 스냅샷

2026-07-14 기준 환경 계약:

- conda 사용 가능 여부는 환경마다 다르므로 `which conda` 또는 `conda info --envs`로 확인한다.
- 현재 Python: 3.12 계열
- 핵심 패키지: `numpy`, `torch 2.8.0+cu128`, `sklearn`, `yaml` 사용 가능
- `matplotlib`은 현재 base 환경에 없음. 기존 runnable unit 다수는 직접 SVG를 쓰므로 필수는 아니다.
- GPU 번호와 여유 상태는 고정 정보가 아니다. 매 실험 직전 다시 조회한다.

최신 상태는 아래 명령으로 다시 확인한다.

```bash
python scripts/check_experiment_environment.py
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
conda info --envs
```

## 운영 원칙

1. 커리큘럼 계약 테스트와 기본 runnable lab는 CPU-safe deterministic으로 유지한다.
2. `fidelity: gpu-capable`은 device 계약을 지원한다는 뜻이지, 과거 GPU 실행이 성공했다는 증명이 아니다.
3. 유휴 GPU를 쓸 때는 명시적으로 `CUDA_VISIBLE_DEVICES=<idle_gpu>`를 설정한다.
4. 큰 checkpoint, raw run log, tensorboard log는 Git에 올리지 않고 `runs/` 또는 외부 artifact store에 둔다.
5. 다시 볼 가치가 있는 요약만 `reports/` 또는 `artifacts/promoted/` 규약에 맞춰 승격한다.

## 권장 실행 예시

### CPU 기준선

```bash
BTB_DEVICE=cpu python 00_foundations/05_gpu_memory_runtime/framework_lab.py
BTB_DEVICE=cpu python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
BTB_DEVICE=cpu python 09_multimodal/01_image_text_retrieval/framework_lab.py
BTB_DEVICE=cpu python 10_vla/01_vision_language_action_grounding/framework_lab.py
```

### 유휴 GPU의 작은 대표 스모크

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> BTB_DEVICE=cuda python 00_foundations/05_gpu_memory_runtime/framework_lab.py
CUDA_VISIBLE_DEVICES=<idle_gpu> BTB_DEVICE=cuda python 05_advanced_nlp_llm/04_instruction_tuning_and_sft/framework_lab.py
CUDA_VISIBLE_DEVICES=<idle_gpu> BTB_DEVICE=cuda python 09_multimodal/01_image_text_retrieval/framework_lab.py
CUDA_VISIBLE_DEVICES=<idle_gpu> BTB_DEVICE=cuda python 10_vla/01_vision_language_action_grounding/framework_lab.py
```

이 명령들은 대형 checkpoint를 다운로드하지 않는 tiny educational lab이다. RLHF reasoning simulation처럼 개념 실습인 unit은 GPU 증명에 사용하지 않는다.

## artifact 검증 계약

실행 후 각 `artifacts/framework-manual/metrics.json`에서 다음을 확인한다.

- `device` 값이 실제 요청과 같은지 확인한다.
- Foundations는 CUDA memory 할당과 runtime이 유한한지 확인한다.
- SFT는 `initial_loss > final_loss`, retrieval·VLA는 `loss_history_head`의 첫 loss가 `loss_history_tail`의 마지막 loss보다 큰지 확인하고, 해당 task metric/failure probe를 함께 본다.
- CPU/GPU 실행의 소수점이 bitwise 동일할 필요는 없지만, 학습 경향과 결과 schema는 같아야 한다.
- artifact는 기본적으로 ignored 실행 증거다. 장기 보존은 명령·GPU·metric 요약만 별도 report에 승격한다.

## 아직 큰 GPU 실험을 자동 실행하지 않는 이유

현재 저장소의 test contract는 CPU-safe deterministic unit을 기준으로 설계되어 있다. 무거운 GPU 실험을 기본 검증에 넣으면 다른 사용자의 GPU 작업과 충돌하거나, Git에 올라가지 않아야 할 큰 산출물이 생길 수 있다. 따라서 기본 구현은 CPU-safe로 검증하고, 유휴 GPU는 명시적 선택 실험에 배정한다.
