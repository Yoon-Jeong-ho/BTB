# 01 ML 환경

ML 트랙은 다른 단계와 충돌하지 않도록 전용 conda 환경 `btb-01-ml` 에서 실행한다. 이제 실행 진입점과 산출물도 모두 `01_ml/` 안에 모아 두기 때문에, 환경 문서도 그 구조에 맞춰 읽으면 된다.

## 왜 분리했는가

- `01_ml`은 scikit-learn / xgboost / torch 비교 실험을 함께 돌리므로 이후 NLP·멀티모달 환경과 충돌할 수 있다.
- GPU 0을 사용하는 비교 실험이 포함되어 있어, 의존성과 CUDA 확인을 한곳에 묶는 편이 안전하다.
- 실험 문서와 artifact를 공부용으로 남기기 위해 `01_ml/<stage>/artifacts/` 구조를 사용한다.

## 실제 생성한 환경

- Conda env name: `btb-01-ml`
- Python: `3.12.4`
- Torch: `2.8.0+cu128`
- 기본 GPU: `CUDA_VISIBLE_DEVICES=0`

## 재현 방법

### 1) 최소 spec으로 만들기

```bash
bash 01_ml/env/create_env.sh
conda activate btb-01-ml
```

### 2) lock 파일로 복원하기

```bash
conda create -n btb-01-ml --file 01_ml/env/conda-linux-64.lock.txt
conda activate btb-01-ml
```

## 실행 명령

### 전체 ML 트랙 실행

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/run_all.py --gpu 0
```

### stage 하나만 실행

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/01_tabular_classification/run_stage.py --gpu 0
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/02_tabular_regression/run_stage.py --gpu 0
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/03_model_selection_and_interpretation/run_stage.py --gpu 0
CUDA_VISIBLE_DEVICES=0 conda run -n btb-01-ml python 01_ml/04_large_scale_tabular/run_stage.py --gpu 0
```

## 구조 메모

- 실행 진입점: `01_ml/run_all.py`
- stage별 코드: `01_ml/<stage>/`
- stage별 artifact: `01_ml/<stage>/artifacts/<run_id>/`
- 환경 spec: `01_ml/env/environment.yml`
- 환경 lock: `01_ml/env/conda-linux-64.lock.txt`

## 주의

- conda env 자체는 저장소 밖 시스템 경로에 둔다.
- Git에는 공부용으로 남길 문서/figure/metrics만 유지하고, 임시 로그/체크포인트는 `.gitignore`로 관리한다.
- 이후 트랙도 같은 방식으로 `env/ + stage-local code + stage-local artifacts` 구조를 따르는 것이 목표다.
