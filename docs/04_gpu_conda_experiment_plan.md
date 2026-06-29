# 04 GPU / Conda Experiment Plan

## 현재 환경 스냅샷

2026-06-24 현재 이 작업 세션에서 확인한 내용:

- conda 사용 가능 여부는 환경마다 다르므로 `which conda` 또는 `conda info --envs`로 확인한다.
- 현재 Python: 3.12 계열
- 핵심 패키지: `numpy`, `torch 2.8.0+cu128`, `sklearn`, `yaml` 사용 가능
- `matplotlib`은 현재 base 환경에 없음. 기존 runnable unit 다수는 직접 SVG를 쓰므로 필수는 아니다.
- GPU 0–3은 사용률이 높고, GPU 4–7은 약 48GB free / 0% utilization로 선택 실험 후보였다.

최신 상태는 아래 명령으로 다시 확인한다.

```bash
python scripts/check_experiment_environment.py
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
conda info --envs
```

## 운영 원칙

1. 커리큘럼 계약 테스트와 기본 runnable lab는 CPU-safe deterministic으로 유지한다.
2. GPU는 `05_advanced_nlp_llm`, `06_training_systems`, `09_multimodal`, `10_vla`의 선택 확장 실험에만 배정한다.
3. 유휴 GPU를 쓸 때는 명시적으로 `CUDA_VISIBLE_DEVICES=<idle_gpu>`를 설정한다.
4. 큰 checkpoint, raw run log, tensorboard log는 Git에 올리지 않고 `runs/` 또는 외부 artifact store에 둔다.
5. 다시 볼 가치가 있는 요약만 `reports/` 또는 `artifacts/promoted/` 규약에 맞춰 승격한다.

## 권장 실행 예시

### VLA toy grounding 확인

```bash
python 10_vla/01_vision_language_action_grounding/scratch_lab.py
python 10_vla/01_vision_language_action_grounding/framework_lab.py
python 10_vla/01_vision_language_action_grounding/analysis.py
```

### 유휴 GPU로 optional heavy lab를 실행할 때

```bash
CUDA_VISIBLE_DEVICES=4 python 09_multimodal/01_image_text_retrieval/framework_lab.py
CUDA_VISIBLE_DEVICES=4 python 05_advanced_nlp_llm/06_rlhf_and_reasoning_rl/framework_lab.py
```

실행 후에는 `nvidia-smi`로 점유를 확인하고, 결과가 재사용 가치가 있을 때만 `reports/`로 요약을 승격한다.

## 아직 큰 GPU 실험을 자동 실행하지 않는 이유

현재 저장소의 test contract는 CPU-safe deterministic unit을 기준으로 설계되어 있다. 무거운 GPU 실험을 기본 검증에 넣으면 다른 사용자의 GPU 작업과 충돌하거나, Git에 올라가지 않아야 할 큰 산출물이 생길 수 있다. 따라서 기본 구현은 CPU-safe로 검증하고, 유휴 GPU는 명시적 선택 실험에 배정한다.
