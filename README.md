# BTB

## NLP 바보에서 박사

이 저장소는 `00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal -> 10_vla` 순서로 올라가면서, foundations/deep-learning core/bridge/applied/systems/frontier/multimodal을 모두 작은 실험과 산출물로 확인하는 한글 우선 학습 저장소다.

핵심 철학은 세 가지다.

1. 쉬운 baseline부터 시작해 더 강한 모델과 운영 습관으로 확장한다.
2. 모든 실험은 `로그`, `결과 figure`, `분석 figure`, `실패 사례`, `summary.md`를 남긴다.
3. Git에는 이해와 재현에 필요한 산출물만 남기고, 큰 가중치는 Hugging Face Hub로 분리한다.

## 10분 시작

**GPU가 없어도 시작할 수 있다.** 첫 단원과 대부분의 기본 실습은 CPU에서 동작하며, GPU는 `gpu-capable` 단원에서만 선택적으로 쓴다. 먼저 현재 환경을 확인하고 텐서 단원 하나를 끝까지 실행해 보는 것이 가장 빠르다.

```bash
# 1) 저장소 루트에서 현재 Python·핵심 패키지·GPU 상태를 확인한다.
python scripts/check_experiment_environment.py

# 2) 첫 단원의 scratch → framework → analysis를 CPU로 실행한다.
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode all --device cpu

# 3) 이번 실행에서 생긴 지표·분석 질문·artifact 링크를 한 장으로 확인한다.
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
```

명령이 끝나면 `00_foundations/01_tensor_shapes/artifacts/summary.md`를 열어 `metrics.json`, 생성된 그림, 분석 질문을 함께 확인한다. 패키지가 없거나 CUDA를 쓸 수 없다는 메시지가 나오면 먼저 GPU 설치를 고치려 하지 말고, [learner preflight](docs/00_learner_preflight.md)의 해당 항목과 CPU 실습을 계속 진행한다.

### 브라우저에서 따라가기

문서·코드·실행 결과·체크리스트를 한 화면에서 보려면 다음 서버를 사용한다.

```bash
python scripts/study_server.py --port 8000 --device auto
# 브라우저에서 http://localhost:8000/web/ 열기
```

| 원하는 일 | 사용할 방법 | 알아둘 점 |
| --- | --- | --- |
| 문서만 읽기 | `python -m http.server 8000` | Python 실행 버튼은 작동하지 않는다. |
| 코드도 실행하기 | `python scripts/study_server.py --port 8000 --device auto` | `cpu-or-cuda` 단원만 유휴 GPU를 자동 탐색한다. |
| GPU를 쓰지 않기 | `--device cpu` | 공유 GPU를 잡지 않는 가장 안전한 기본값이다. |
| 특정 GPU를 명시하기 | `--device cuda --gpu-index 0` | 본인이 사용할 수 있는 유휴 GPU인지 먼저 확인한다. |

웹사이트는 체크 표시만으로 완료 처리하지 않는다. 필수 체크포인트, 성공한 실행 artifact, 현재 퀴즈 답변, 짧은 회고 메모가 모두 있을 때 숙달 증거로 표시된다. Node/Playwright는 학습에 필요하지 않고, 웹사이트를 수정한 뒤 `npm run qa:web`으로 화면 QA를 할 때만 필요하다.

## 내 시작점 고르기

| 현재 상태 | 추천 첫 경로 | 첫 목표 |
| --- | --- | --- |
| Python·행렬·tensor가 낯설다 | [preflight](docs/00_learner_preflight.md) → `00_foundations/01` → `00_foundations/03` | shape, loss, gradient를 말과 코드로 연결한다. |
| sklearn은 써 봤지만 딥러닝은 처음이다 | `01_ml/01` → `02_deep_learning/01` → `02_deep_learning/04` | baseline·metric·오류 분석을 training loop와 attention으로 옮긴다. |
| LLM만 빠르게 이해하고 싶다 | `00_foundations` 핵심 → `03_nlp_bridge` → `05_advanced_nlp_llm` | token/embedding/attention을 SFT·RAG·alignment와 연결한다. |
| Multimodal/VLA가 목표다 | LLM 경로 → `08_multimodal_bridge` → `09_multimodal` → `10_vla` | retrieval·grounding·safety gate를 순서대로 확인한다. |
| 분산 학습·논문 재현이 필요하다 | 위 경로 뒤 `06_training_systems`, `07_frontier_labs` | `torchrun`, shard, 재현성, capstone으로 확장한다. |

`06_training_systems`와 `07_frontier_labs`는 초심자에게 **선택형 사이드카**다. 폴더 순서는 전체 지도를 보여 주지만, LLM·Multimodal·VLA 입문 전에 반드시 끝낼 관문은 아니다.

## 학습 순서

1. [00_foundations](00_foundations/README.md): 공통 수치/텐서/실행 감각을 먼저 고정한다.
2. [01_ml](01_ml/README.md): baseline, metric, error analysis, experiment discipline을 익힌다.
3. [02_deep_learning](02_deep_learning/README.md): perceptron부터 transformer·generative model까지 딥러닝 모델 패밀리를 한 번 정리한다.
4. [03_nlp_bridge](03_nlp_bridge/README.md): tokenization, embedding, sequence modeling, attention을 통해 DL에서 NLP로 넘어가는 입력/표현 다리를 놓는다.
5. [04_nlp](04_nlp/README.md): text classification, NER, MRC 같은 applied NLP core를 실습한다.
6. [05_advanced_nlp_llm](05_advanced_nlp_llm/README.md): pretraining 이후의 advanced NLP/LLM, instruction tuning, preference optimization, RAG, alignment를 묶어 본다.
7. [06_training_systems](06_training_systems/README.md): distributed training, sharding, parallelism, checkpointing, failure recovery를 다룬다.
8. [07_frontier_labs](07_frontier_labs/README.md): paper reproduction, capstone, agentic experiment를 수행하는 연구형 실습 구간이다.
9. [08_multimodal_bridge](08_multimodal_bridge/README.md): contrastive alignment와 image-text shared representation으로 멀티모달 연결 다리를 만든다.
10. [09_multimodal](09_multimodal/README.md): retrieval, captioning, VQA를 중심으로 multimodal applied track을 실습한다.
11. [10_vla](10_vla/README.md): vision-language-action grounding, action token, safety gate를 통해 VLA 입구를 만든다.

무기초에서 LLM/RLHF/Multimodal/VLA까지 빠르게 올라가는 루트에서는 `06_training_systems`와 `07_frontier_labs`를 **선택형 사이드카**로 두고 나중으로 미뤄도 된다. 폴더 번호와 canonical 전체 순서는 바꾸지 않는다. 먼저 `00→05`로 언어 모델의 이론·코드·실험 습관을 만들고, `08_multimodal_bridge -> 09_multimodal -> 10_vla`로 넘어간 뒤, 큰 GPU 실험·분산 운영·논문 재현이 필요해질 때 `06_training_systems`와 `07_frontier_labs`를 optional capstone sandbox로 되돌아본다. `10_vla/01_vision_language_action_grounding`은 실제 로봇 제어 전체가 아니라 VLA grounding entry point이므로, 행동 토큰과 safety gate의 최소 감각을 잡는 입구로 읽는다.

현재 `docs/curriculum_status.json`에 선언된 모든 unit은 `runnable` 상태다. 그래도 실행 전에는 manifest와 각 track README의 status table을 함께 확인해, 어떤 산출물과 분석 질문을 남겨야 하는지 먼저 읽는 것을 원칙으로 한다.

현재 경로 기준으로 `03_nlp_bridge -> 04_nlp`, `08_multimodal_bridge -> 09_multimodal`이 bridge/applied 흐름을 담당한다. 과거 경로에서 현재 경로로 바뀐 자세한 대응표는 [docs/03_track_migration_map.md](docs/03_track_migration_map.md)에서만 historical reference로 다룬다.

시작 전 준비도는 [docs/00_learner_preflight.md](docs/00_learner_preflight.md), 전체 프로그램 개요는 [docs/00_program_map.md](docs/00_program_map.md), 추천 학습 동선은 [docs/02_study_guide.md](docs/02_study_guide.md), 경로 변경 안내는 [docs/03_track_migration_map.md](docs/03_track_migration_map.md), 실험 운영 규칙은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md)에 정리했다.

## 저장소 구조

```text
BTB/
├── 00_foundations/             # 공통 수치/텐서/런타임 기초
├── shared/                    # 번호 없는 공통 규약, 템플릿, 요약 포맷
├── 01_ml/                      # baseline, metric, 해석 discipline
├── 02_deep_learning/           # 딥러닝 모델 패밀리 학습
├── 03_nlp_bridge/              # DL -> NLP 입력/표현 브리지
├── 04_nlp/                     # applied NLP core
├── 05_advanced_nlp_llm/        # pretraining 이후 고급 NLP/LLM
├── 06_training_systems/        # distributed / large-model systems
├── 07_frontier_labs/           # reproduction, capstone, agentic labs
├── 08_multimodal_bridge/       # multimodal 연결 브리지
├── 09_multimodal/              # multimodal applied track
├── 10_vla/                     # vision-language-action grounding
├── web/                        # 정적 커리큘럼 웹사이트(localStorage 진행 체크)
├── data/                       # raw/interim/processed/external 설명용 구조
├── runs/                       # 서버/로컬의 비정제 실행 산출물(기본 ignore)
├── reports/                    # Git에 남길 승격된 실험 결과
├── artifacts/                  # 모델 가중치/레지스트리 규칙
├── docs/                       # 프로그램 개요, 학습 가이드, 마이그레이션 노트
└── scripts/                    # 학습/평가/검증 스크립트 인터페이스 규약
```

최상위 폴더와 단계 폴더에 인덱스를 붙여 학습 순서가 정렬되도록 유지한다. 문서는 한글 우선을 기본으로 하고, 코드/파일명/핵심 technical term만 영어를 유지한다. 실험도 같은 방식으로 `01_...`, `02_...` 순서를 유지한다.

## 실험 산출물 규약

모든 실험은 최소한 아래 산출물을 남긴다.

- `config.yaml`: 하이퍼파라미터, 데이터 버전, seed
- `metrics.json`: 주요 지표
- `summary.md`: 한 번에 읽히는 실험 요약
- `figures/results/`: 학습 곡선, confusion matrix, retrieval 성능, caption 예시 등
- `figures/analysis/`: 에러 분석, slice 분석, feature importance, failure case panel 등
- `predictions/`: 샘플 예측 결과
- `model_card.md`: 승격할 가치가 있는 모델이면 작성

상세 규약은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md)를 따른다.

## Git / 서버 / Hugging Face 운영 원칙

- 로컬에서는 문서 정리, 분석, 소규모 실험, 결과 선별을 담당한다.
- 서버에서는 실제 학습과 대량 로그/체크포인트 생성을 담당한다.
- `runs/`는 기본적으로 Git에서 제외한다.
- 팀이 다시 볼 가치가 있는 결과만 `reports/`와 `artifacts/promoted/`로 승격한다.
- 작은 가중치는 `artifacts/promoted/` 아래에서 Git LFS로 관리할 수 있다.
- 큰 가중치는 Hugging Face Hub에 업로드하고, 링크와 커밋 정보를 [artifacts/MODEL_REGISTRY.md](artifacts/MODEL_REGISTRY.md)에 기록한다.

Hugging Face 업로드와 Git LFS 관련 규칙은 루트의 `.gitignore`, `.gitattributes`, 그리고 [artifacts/README.md](artifacts/README.md)에 정리했다.

## 한 단원을 끝내는 방법

1. [preflight](docs/00_learner_preflight.md)와 [study guide](docs/02_study_guide.md)에서 현재 시작점과 다음 트랙을 고른다. 전체 역할은 [program map](docs/00_program_map.md)에서 확인한다.
2. 단원의 `README.md`에서 목표·선행지식·예상 시간·실습 성격을 읽고, `THEORY.md`와 `PREREQS.md`를 필요한 만큼만 먼저 본다.
3. `scratch → framework → analysis` 순서로 실행한다. terminal에서는 `scripts/run_lesson.py`, 브라우저에서는 실행 버튼을 사용한다. ML real-data 단원은 `run_stage.py` 하나가 전체 실험 진입점이다.
4. 실행 뒤 `metrics.json` 숫자 하나, 생성된 figure 하나, failure case 하나를 보고 단원의 분석 질문에 자신의 말로 답한다. `build_lesson_report.py`가 이 증거를 `artifacts/summary.md`로 묶어 준다.
5. 웹을 쓴다면 체크포인트·퀴즈·회고 메모까지 남겨 숙달 증거를 완성한다. terminal 중심이라면 [run summary template](shared/templates/run_summary_template.md)으로 같은 내용을 남긴다.
6. 다시 볼 가치가 있는 결과만 [reports/README.md](reports/README.md) 규칙에 맞게 승격한다. 모델 가중치와 대량 실행 로그는 Git에 바로 넣지 않는다.

자세한 웹 동작은 [web/README.md](web/README.md), 모든 실행/검증 명령은 [scripts/README.md](scripts/README.md), GPU/conda 실험 순서는 [docs/04_gpu_conda_experiment_plan.md](docs/04_gpu_conda_experiment_plan.md)를 따른다.

## 참고 자료

사용한 공식 사이트, 벤치마크, GitHub, 문서는 [docs/90_references.md](docs/90_references.md)에 모아 두었다.
