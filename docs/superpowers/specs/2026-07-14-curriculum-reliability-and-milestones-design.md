# BTB 커리큘럼 신뢰성 및 실전 마일스톤 설계

## 배경

BTB는 48개 runnable unit에 한글 우선 문서, scratch/framework/analysis/reflection 흐름, 산출물과 실패 사례 중심 학습 철학을 갖추고 있다. 기준선 검증은 304 tests와 88 subtests가 통과한다. 그러나 현재 `runnable`은 CPU toy simulation, 실제 framework 계산, real-data 실험, GPU 검증을 구분하지 않으며 웹·CLI·report가 서로 다른 실행 계약을 사용한다.

이 설계는 기존 toy lab을 삭제하거나 전체 커리큘럼을 무겁게 바꾸지 않는다. 대신 모든 단원의 수준과 완료 기준을 정직하게 드러내고, ML→LLM→Multimodal→VLA의 대표 전환점에서 실제 계산 및 GPU 증거를 남긴다.

## 목표

1. manifest에 runnable로 선언된 모든 단원이 공통 parser와 runner에서 실제로 열리고 실행될 수 있게 한다.
2. 단원 수준을 `concept-toy`, `framework-toy`, `real-data`, `gpu-capable`로 구분한다.
3. CPU/GPU 선택이 웹 서버 표시부터 artifact의 `device` 값까지 일치하게 한다.
4. 실행 성공, 분석, 퀴즈, 회고를 학습 숙달 증거로 분리해 저장한다.
5. 초심자가 진입 준비도와 선택형 Systems/Frontier 경로를 오해하지 않게 한다.
6. 새 의존성 없이 현재 설치된 Python/Torch 도구로 대표 실전 마일스톤을 검증한다.

## 비목표

- 48개 toy lab을 모두 대규모 pretrained-model 학습으로 대체하지 않는다.
- `10_vla`를 실제 로봇 제어 전체 과정이라고 표현하지 않는다.
- 대형 모델 가중치나 외부 데이터셋 전체를 Git에 저장하지 않는다.
- 기존 artifact 경로와 웹 localStorage 데이터를 불필요하게 깨지 않는다.

## 접근 비교

### A. 계약 정비만

parser, runner, report, GPU 전달만 수정한다. 위험과 변경량은 작지만 실제 학습 전이를 강화하지 못한다.

### B. 전체 콘텐츠 재작성

모든 unit을 실제 데이터/API 중심으로 다시 만든다. 깊이는 늘지만 초심자용 작은 숫자 계단을 잃고 변경 범위가 지나치게 넓다.

### C. 계약 정비 + 대표 실전 마일스톤

기존 toy lab을 보존하고 수준을 명시하며, 주요 전환점에 실제 실행 증거를 추가한다. 신뢰성·학습 전이·변경 위험의 균형이 가장 좋아 이 접근을 채택한다.

## 설계

### 1. 단일 lesson metadata 계약

`scripts/_lesson_metadata.py`를 canonical loader로 사용하고 catalog builder의 중복 parser를 제거한다. loader는 기존 top-level scalar/list와 현재 저장소에 존재하는 한 단계 nested mapping을 안전하게 파싱한다.

모든 manifest unit에 대해 다음을 검증한다.

- `objective`, `prereqs`, `key_terms`, `required_outputs`, `analysis_questions`
- unit 폴더와 runnable entrypoint 존재
- metadata parser 호환성
- 선언된 prerequisite unit/document의 존재
- 학습 수준과 compute profile의 명시

학습 수준 필드는 다음 의미를 갖는다.

- `concept-toy`: 개념을 수작업 숫자나 deterministic simulation으로 확인
- `framework-toy`: 실제 framework tensor/model/update를 작은 입력에서 실행
- `real-data`: 외부 또는 실제 형태 데이터와 evaluation contract 사용
- `gpu-capable`: `BTB_DEVICE`를 존중하는 작은 CPU/CUDA 실습. 과거 검증을 과장하지 않고 현재 실행 artifact의 device로 검증함

### 2. 실행 및 산출물 계약

`run_lesson.py`는 기존 scratch/framework 사용법을 보존하고 다음을 추가한다.

- `analysis` mode
- `all` mode: scratch → framework → analysis
- `--device auto|cpu|cuda`
- 실행 전 선택된 unit/mode/device 표시
- 실행 후 실제 생성된 artifact와 metadata objective 요약

`BTB_DEVICE`는 framework lab이 읽는 공통 device contract다. `cpu`는 CUDA를 숨기고, `cuda`는 CUDA가 없을 때 명시적으로 실패하며, `auto`는 사용 가능한 CUDA 또는 CPU를 선택한다.

`build_lesson_report.py`는 알 수 없는 output label을 조용히 무시하지 않는다. 표준 artifact 이름은 실제 경로로 정규화하고, 해석 실행 증거는 static `analysis.md`가 아니라 generated observed report를 우선한다. report에는 metric key뿐 아니라 핵심 값, device, artifact 링크, 분석 질문을 포함한다.

### 3. 학습자 경로와 준비도

기본 core path는 다음과 같이 설명한다.

`Foundations → ML minimum → DL core → NLP → LLM → Multimodal bridge → Multimodal → VLA entry`

`06_training_systems`와 `07_frontier_labs`는 GPU/분산/연구 재현이 필요한 시점의 optional sidecar로 표시한다.

기존 첫 foundations unit 앞에는 별도 대규모 track 대신 가벼운 learner preflight를 제공한다. Python, tensor shape, 행렬곱, 확률/metric, CLI, PyTorch/CUDA 준비 상태를 점검하고 부족한 항목별 시작 문서를 안내한다. 진단 결과는 학습을 차단하지 않는다.

### 4. 대표 실전 마일스톤

새 의존성을 추가하지 않고 기존 unit을 강화한다.

- Foundations: GPU memory/runtime lab에서 실제 CUDA memory/runtime 증거
- ML: 기존 real-data tabular stage의 artifact 계약 확인
- LLM: 작은 Torch model의 실제 forward/loss/backward/update와 device 기록
- Systems: 최소 2-process `torchrun` smoke 또는 환경 불가 시 명확한 optional status
- Multimodal: dual encoder의 CPU/CUDA 학습 및 retrieval metric parity
- VLA: action/safety policy의 CPU/CUDA 학습 및 failure probe 보존

각 GPU 실험은 실행 직전에 `nvidia-smi`로 idle 상태를 다시 확인하고, 작은 모델과 짧은 epoch만 사용한다.

### 5. 웹 숙달 증거

기존 수동 진행 상태는 보존하되 `done`과 별도로 검증된 숙달 증거를 계산한다.

- 필수 읽기 checkpoint
- 성공한 Python 실행과 artifact/device 정보
- 비어 있지 않은 서술 답변
- 단원 퀴즈 제출 상태
- 비어 있지 않은 회고/다음 가설

실행 증거는 localStorage 사용자 프로필에 저장해 reload 뒤에도 남긴다. 저장하는 값은 command 전체나 민감한 환경 변수가 아니라 unit path, resource, exit code, device, timestamp, artifact 요약이다.

### 6. GPU 안전성

- 명시한 `gpu_index`도 idle threshold를 통과해야 auto 선택한다.
- CUDA 장치가 없으면 `--device cuda`는 CUDA로 가장하지 않고 actionable error를 낸다.
- 동시에 중복 실행되는 요청을 제한한다.
- UI runner device와 artifact `device` 불일치를 테스트한다.

## 오류 처리

- schema/parser 오류에는 파일과 줄 번호를 포함한다.
- required artifact가 없으면 실행해야 할 mode를 안내한다.
- optional heavy lab에 필요한 package/API가 없으면 toy lab 결과와 구분된 skip 이유를 남긴다.
- GPU가 바쁘거나 없으면 auto는 CPU로 fallback하고 강제 CUDA는 실패한다.
- 학습자의 수동 메모와 진행률은 schema migration 시 보존한다.

## 테스트 전략

1. 모든 manifest unit metadata를 canonical loader로 파싱한다.
2. nested mapping과 기존 scalar/list 형식의 regression test를 추가한다.
3. runner의 scratch/framework/analysis/all/device 계약을 테스트한다.
4. report가 미지원 output을 무시하지 않고 실제 artifact를 검증하는지 테스트한다.
5. study server의 busy pinned GPU, no-GPU forced CUDA, run evidence를 테스트한다.
6. 웹에서 빈 서술 답변 거부, 실행 증거 persistence, 숙달 상태 계산을 테스트한다.
7. 전체 Python test suite와 JS syntax/Playwright QA를 실행한다.
8. 새로 확인한 idle GPU에서 Foundations, Multimodal, VLA 스모크를 실행하고 artifact device와 CPU/GPU metric parity를 확인한다.

## 완료 기준

- manifest의 모든 unit가 canonical metadata audit를 통과한다.
- 기존 304 tests와 88 subtests를 포함한 전체 회귀가 통과한다.
- catalog 재생성이 clean diff를 만든다.
- 대표 CPU 및 GPU 실험이 성공하고 device/artifact 증거가 남는다.
- 웹 QA에서 학습 진행, 실행, 숙달 증거가 reload 후 유지된다.
- 문서가 toy와 authentic/GPU-validated 범위를 과장 없이 설명한다.
