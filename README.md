# BTB

## NLP 바보에서 박사

이 저장소는 `00_foundations -> 01_ml -> 02_deep_learning -> 03_nlp_bridge -> 04_nlp -> 05_advanced_nlp_llm -> 06_training_systems -> 07_frontier_labs -> 08_multimodal_bridge -> 09_multimodal` 순서로 올라가면서, foundations/deep-learning core/bridge/applied/systems/frontier/multimodal을 모두 작은 실험과 산출물로 확인하는 한글 우선 학습 저장소다.

핵심 철학은 세 가지다.

1. 쉬운 baseline부터 시작해 더 강한 모델과 운영 습관으로 확장한다.
2. 모든 실험은 `로그`, `결과 figure`, `분석 figure`, `실패 사례`, `summary.md`를 남긴다.
3. Git에는 이해와 재현에 필요한 산출물만 남기고, 큰 가중치는 Hugging Face Hub로 분리한다.

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

현재 `docs/curriculum_status.json`에 선언된 모든 unit은 `runnable` 상태다. 그래도 실행 전에는 manifest와 각 track README의 status table을 함께 확인해, 어떤 산출물과 분석 질문을 남겨야 하는지 먼저 읽는 것을 원칙으로 한다.

현재 경로 기준으로 `03_nlp_bridge -> 04_nlp`, `08_multimodal_bridge -> 09_multimodal`이 bridge/applied 흐름을 담당한다. 과거 경로에서 현재 경로로 바뀐 자세한 대응표는 [docs/03_track_migration_map.md](docs/03_track_migration_map.md)에서만 historical reference로 다룬다.

전체 프로그램 개요는 [docs/00_program_map.md](docs/00_program_map.md), 추천 학습 동선은 [docs/02_study_guide.md](docs/02_study_guide.md), 경로 변경 안내는 [docs/03_track_migration_map.md](docs/03_track_migration_map.md), 실험 운영 규칙은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md)에 정리했다.

## 저장소 구조

```text
BTB/
├── 00_foundations/             # 공통 수치/텐서/런타임 기초
├── 00_shared/                  # 공통 규약, 템플릿, 요약 포맷
├── 01_ml/                      # baseline, metric, 해석 discipline
├── 02_deep_learning/           # 딥러닝 모델 패밀리 학습
├── 03_nlp_bridge/              # DL -> NLP 입력/표현 브리지
├── 04_nlp/                     # applied NLP core
├── 05_advanced_nlp_llm/        # pretraining 이후 고급 NLP/LLM
├── 06_training_systems/        # distributed / large-model systems
├── 07_frontier_labs/           # reproduction, capstone, agentic labs
├── 08_multimodal_bridge/       # multimodal 연결 브리지
├── 09_multimodal/              # multimodal applied track
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

## 시작 순서

1. [docs/00_program_map.md](docs/00_program_map.md)로 전체 트랙의 역할 경계를 먼저 본다.
2. [docs/02_study_guide.md](docs/02_study_guide.md)에서 자신에게 맞는 학습 동선을 고른다.
3. [00_foundations/README.md](00_foundations/README.md)와 [01_ml/README.md](01_ml/README.md)로 공통 기초와 baseline 운영 습관을 먼저 다진다.
4. 각 track에 들어가기 전에는 먼저 [docs/curriculum_status.json](docs/curriculum_status.json)에서 `runnable` 상태와 unit 목록을 확인하고, 해당 track README의 status table을 보조 설명으로 함께 읽는다.
5. 실험을 돌릴 때는 [00_shared/templates/run_summary_template.md](00_shared/templates/run_summary_template.md) 형식으로 요약을 남기고, 다시 볼 가치가 있는 결과만 [reports/README.md](reports/README.md) 규칙에 맞게 승격한다.

## 참고 자료

사용한 공식 사이트, 벤치마크, GitHub, 문서는 [docs/90_references.md](docs/90_references.md)에 모아 두었다.
