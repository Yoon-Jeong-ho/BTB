# 00 Learner Preflight

## 목적

이 사전진단은 BTB를 시작하기 전에 **Python / CLI, 수학, 확률 / metric, PyTorch / GPU**의 현재 준비도를 짧게 확인하는 체크리스트다. 시험이나 입장 제한이 아니다. 모르는 항목은 실패가 아니라 먼저 밟을 계단을 고르는 신호다.

- 답을 외우지 말고, 명령을 직접 실행하거나 한두 문장으로 설명해 본다.
- 한 영역에서 두 항목 이상 막히면 아래의 추천 경로를 먼저 따른다.
- GPU가 없거나 CUDA가 동작하지 않아도 core curriculum은 CPU로 시작할 수 있다.
- 각 `lesson.yaml`의 `fidelity`, `difficulty`, `estimated_minutes`, `compute`를 보고 실습의 깊이와 필요한 자원을 먼저 확인한다.

## 메타데이터 읽는 법

### `fidelity`

| 값 | 의미 | 기대해도 되는 것 |
| --- | --- | --- |
| `concept-toy` | 작은 수치나 deterministic simulation으로 개념을 확인 | 알고리즘/시스템의 인과관계와 산출물 형식 |
| `framework-toy` | PyTorch 같은 실제 framework tensor/model 연산을 작은 synthetic 입력에서 실행 | forward, loss, update, shape trace |
| `real-data` | 공개 real dataset과 evaluation contract를 사용 | split, baseline, metric, error analysis |
| `gpu-capable` | 같은 작은 실습이 `BTB_DEVICE`에 따라 CPU/CUDA에서 실행 | 실제 device 기록과 CPU fallback; 대규모 학습이나 과거 GPU 검증을 뜻하지 않음 |

### 나머지 필드

- `difficulty`: `beginner`, `intermediate`, `advanced` 중 진입 난이도다. 지능이나 성취도를 평가하는 등급이 아니다.
- `estimated_minutes`: 읽기, 실행, 분석 질문, 회고까지 한 번 수행하는 대략적인 시간이다.
- `compute`: `cpu`, `cpu-or-cuda`, `optional-multiprocess` 중 기본 실행 조건이다. `optional-multiprocess`는 단일 CPU 개념 실습 뒤 별도 다중 프로세스 확장을 시도할 수 있음을 뜻한다.

## 1. Python / CLI

저장소 루트에서 다음을 확인한다.

```bash
python --version
python -c "from pathlib import Path; print(Path.cwd().name)"
python scripts/run_lesson.py --help
```

자가 점검:

- [ ] 현재 작업 디렉터리와 상대 경로의 차이를 설명할 수 있다.
- [ ] `python path/to/file.py --flag value` 형식에서 script, option, value를 구분할 수 있다.
- [ ] list, dict, function argument, `for` loop를 읽을 수 있다.
- [ ] traceback의 마지막 줄에서 예외 종류와 메시지를 찾을 수 있다.
- [ ] JSON과 YAML 파일을 열고 key/value를 찾을 수 있다.

막혔다면:

1. Python의 변수, list/dict, 함수, import, 파일 경로를 짧게 복습한다.
2. 터미널에서 `pwd`, `ls`, `cd`, `python --help`를 직접 사용해 본다.
3. BTB에서는 `00_foundations/01_tensor_shapes`의 README와 `scratch_lab.py`를 나란히 읽으며 시작한다.

## 2. 수학

종이에 shape를 적고 답해 본다.

- [ ] `(batch=4, features=3)` 행렬과 `(features=3, classes=2)` 행렬을 곱한 결과 shape가 `(4, 2)`임을 설명할 수 있다.
- [ ] vector, matrix, scalar를 구분할 수 있다.
- [ ] 평균, 분산, 표준편차가 각각 무엇을 요약하는지 말할 수 있다.
- [ ] 함수의 기울기 부호가 입력을 어느 방향으로 움직여야 하는지 알려 준다는 직관이 있다.
- [ ] dot product가 두 벡터의 정렬 정도와 연결된다는 직관이 있다.

막혔다면:

1. `00_foundations/01_tensor_shapes`를 먼저 실행한다.
2. activation/loss가 낯설면 `00_foundations/02_activation_and_loss`로 간다.
3. gradient가 낯설면 `00_foundations/03_gradients_and_backpropagation`에서 finite difference와 autograd를 비교한다.
4. 세 단원의 분석 질문에 답한 뒤 `01_ml`로 이동한다.

## 3. 확률 / metric

다음 질문에 계산 또는 문장으로 답해 본다.

- [ ] 확률 값이 `0~1` 범위에 있고 분류 확률의 합이 1이 되는 이유를 설명할 수 있다.
- [ ] train/validation/test split의 역할을 구분할 수 있다.
- [ ] accuracy가 높은데 minority class recall이 낮을 수 있는 예를 들 수 있다.
- [ ] classification의 F1과 regression의 MAE/RMSE가 서로 다른 질문에 답한다는 것을 안다.
- [ ] 평균 하나만 보지 않고 class/slice/failure case를 함께 봐야 하는 이유를 설명할 수 있다.

막혔다면:

1. `01_ml/01_tabular_classification`에서 majority baseline, confusion matrix, error slice를 먼저 본다.
2. 연속값 예측 지표가 약하면 `01_ml/02_tabular_regression`에서 residual, MAE, RMSE를 비교한다.
3. validation/test 경계가 약하면 `01_ml/03_model_selection_and_interpretation`로 넘어가기 전에 앞 두 단원의 회고를 다시 쓴다.

## 4. PyTorch / GPU

먼저 import와 tensor 연산을 확인한다.

```bash
python - <<'PY'
import torch
x = torch.tensor([[1.0, 2.0]])
layer = torch.nn.Linear(2, 1)
y = layer(x)
print({"torch": torch.__version__, "shape": list(y.shape), "cuda": torch.cuda.is_available()})
PY
```

자가 점검:

- [ ] tensor의 `shape`, `dtype`, `device`를 출력할 수 있다.
- [ ] model parameter, forward, loss, `backward()`, optimizer `step()`의 순서를 설명할 수 있다.
- [ ] `model.to(device)`만 하고 입력 tensor를 옮기지 않으면 device mismatch가 날 수 있음을 안다.
- [ ] CUDA 사용 가능 여부와 GPU 여유 상태는 서로 다른 조건임을 안다.
- [ ] CPU 결과와 GPU 결과가 bitwise 동일하지 않아도 metric/학습 경향을 비교할 수 있음을 안다.

CUDA를 사용할 계획일 때만 추가 확인한다.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
BTB_DEVICE=cpu python scripts/run_lesson.py --unit 00_foundations/05_gpu_memory_runtime --mode framework --device cpu
```

- `torch.cuda.is_available()`이 `False`면 CPU로 계속 학습한다. CUDA 설치 변경은 현재 환경 관리자와 확인한 뒤 별도로 진행한다.
- `nvidia-smi`에 GPU가 보여도 다른 사용자가 메모리나 연산을 쓰는 중이면 점유하지 않는다.
- `gpu-capable` 단원도 작은 교육용 실험이다. 대형 모델을 내려받거나 장시간 학습한다는 의미가 아니다.
- 강제 CUDA 실행은 idle GPU를 실행 직전에 다시 확인한다. 웹 서버는 `python scripts/study_server.py --device cuda --gpu-index <index>`, CLI runner는 `CUDA_VISIBLE_DEVICES=<index> python scripts/run_lesson.py ... --device cuda`를 사용한다.

막혔다면:

1. tensor/device가 낯설면 `00_foundations/01_tensor_shapes`를 CPU로 실행한다.
2. loss/update가 낯설면 `00_foundations/02_activation_and_loss`와 `03_gradients_and_backpropagation`을 실행한다.
3. training loop가 낯설면 `02_deep_learning/01_perceptron_and_mlp`를 먼저 완료한다.
4. GPU가 없으면 `compute: cpu`와 `cpu-or-cuda` 단원을 CPU로 진행하고, `06_training_systems`의 다중 프로세스 확장은 나중으로 미룬다.

## 진단 결과별 추천 경로

| 관찰 결과 | 권장 시작점 | 다음 확인점 |
| --- | --- | --- |
| Python / CLI에서 두 항목 이상 막힘 | Python/터미널 기초 복습 → `00_foundations/01_tensor_shapes` | scratch 실행과 JSON 산출물 읽기 |
| shape/행렬곱/gradient가 약함 | `00_foundations/01~03` | framework 결과와 손계산 비교 |
| metric/split/error analysis가 약함 | `01_ml/01~02` | baseline 대비 개선과 실패 slice 설명 |
| PyTorch training loop가 약함 | `02_deep_learning/01_perceptron_and_mlp` | forward → loss → backward → step 설명 |
| NLP/LLM이 목표이고 위 항목이 준비됨 | core path의 `03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm` | 생성 bridge와 RL primer 확인 |
| Multimodal/VLA가 목표임 | LLM core 뒤 `08_multimodal_bridge -> 09_multimodal -> 10_vla` | grounding failure와 safety gate 분석 |
| 분산/GPU 운영이 실제로 필요함 | core path 옆 `06_training_systems` 선택형 사이드카 | optional multiprocess smoke와 profile |
| 논문 재현/capstone 문제를 이미 고름 | core 결과를 들고 `07_frontier_labs` 선택형 사이드카 | claim/evidence matrix와 재현 범위 |

## 시작 결정

- **준비됨:** 각 영역에서 4개 이상 설명할 수 있으면 [02 Study Guide](02_study_guide.md)의 core path를 시작한다.
- **부분 준비:** 약한 영역에 해당하는 Foundations/ML/DL 단원만 먼저 수행하고 다시 점검한다.
- **GPU만 미준비:** 학습을 멈추지 않는다. CPU core를 진행하고 `gpu-capable` 단원의 CUDA 비교만 나중에 수행한다.
- **Systems/Frontier가 당장 불필요:** 폴더 번호는 canonical map을 위해 그대로 두되, 두 트랙은 선택형 사이드카로 미룬다.
