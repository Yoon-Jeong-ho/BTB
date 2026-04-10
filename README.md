# BTB

## NLP 바보에서 박사

BTB는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가면서, **읽고 끝내지 않고 작은 실험과 분석으로 확인하는 한글 중심 학습 저장소**다.

핵심 철학은 네 가지다.

1. 쉬운 베이스라인부터 시작한다.
2. 모든 단위는 `이론 -> 작은 구현 -> 실행 결과 -> figure -> 분석 -> 회고` 흐름을 남긴다.
3. Git에는 이해와 재현에 필요한 산출물을 남기고, 큰 가중치는 Hugging Face Hub로 분리한다.
4. 특히 `00_foundations + 02_nlp_bridge`를 **딥러닝 코어 축**으로 보고, activation / loss / gradient / attention / runtime 감각을 먼저 굳힌다. `04_multimodal_bridge`는 그 다음 확장 브리지로 본다.

## 지금 바로 공부를 시작할 때

1. [docs/02_study_guide.md](docs/02_study_guide.md) 에서 **한 unit씩 보는 순서**를 먼저 확인한다.
2. 전체 로드맵이 필요하면 [docs/00_program_map.md](docs/00_program_map.md) 를 본다.
3. foundations/bridge/applied unit는 기본적으로 `README -> THEORY -> (있으면) PREREQS -> scratch_lab/framework_lab -> analysis -> reflection` 순서로 읽고, `01_ml` stage는 `README -> THEORY -> 최신 artifact README` 순서로 읽는다.
4. 실습 실행 규칙과 산출물 계약은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md) 를 따른다.

## 추천 공부 루트

### A. 전체 커리큘럼을 순서대로 가는 기본 루트

1. [00_foundations](00_foundations/README.md): 텐서, activation, gradient, regularization, runtime 같은 공통 기초를 먼저 다진다.
2. [01_ml](01_ml/README.md): 표형 데이터 실험 discipline과 baseline 해석을 익힌다.
3. [02_nlp_bridge](02_nlp_bridge/README.md): tokenization, embedding, attention, transformer block 감각을 연결한다.
4. [03_nlp](03_nlp/README.md): `text classification -> NER -> MRC` 순서로 NLP 적용 실습을 진행한다.
5. [04_multimodal_bridge](04_multimodal_bridge/README.md): contrastive alignment와 retrieval vs generation 차이를 먼저 다진다.
6. [05_multimodal](05_multimodal/README.md): `retrieval -> captioning -> VQA` 순서로 멀티모달 적용 실습을 진행한다.

### B. 딥러닝 코어를 먼저 굳히는 압축 루트

딥러닝 기초가 비어 있다고 느껴지면 아래 순서가 가장 빠르다.

1. `00_foundations/01_tensor_shapes`
2. `00_foundations/02_activation_and_loss`
3. `00_foundations/03_gradients_and_backpropagation`
4. `00_foundations/04_regularization_and_normalization`
5. `00_foundations/05_gpu_memory_runtime`
6. `02_nlp_bridge/01_tokenization_and_embeddings`
7. `02_nlp_bridge/02_attention_and_transformer_block`

이 루트를 끝내면 activation, loss, gradient, backprop, attention, mask, runtime을 한 줄로 설명할 수 있는 상태를 목표로 한다. `04_multimodal_bridge/01_contrastive_alignment` 는 이 다음에 붙이는 선택 확장이다.

## 저장소 구조

```text
BTB/
├── 00_foundations/             # 딥러닝 공통 기초 트랙
├── 00_shared/                  # 공통 규약, 템플릿
├── 01_ml/                      # 실험 discipline과 tabular ML 트랙
├── 02_nlp_bridge/              # 딥러닝 -> NLP 연결 브리지
├── 03_nlp/                     # NLP 적용 트랙
├── 04_multimodal_bridge/       # NLP -> 멀티모달 연결 브리지
├── 05_multimodal/              # 멀티모달 적용 트랙
├── data/                       # raw/interim/processed/external 설명용 구조
├── runs/                       # 서버/로컬의 비정제 실행 산출물(기본 ignore)
├── reports/                    # Git에 남길 승격된 실험 결과
├── artifacts/                  # 모델 가중치/레지스트리 규칙
├── docs/                       # 로드맵, 공부 가이드, 운영 문서, 참고 자료
└── scripts/                    # lesson runner, report builder, link checker
```

최상위 폴더와 각 단계 폴더에 인덱스를 붙여 정렬이 무너지지 않게 했다. 문서는 한글 우선으로 쓰고, 코드/파일명/핵심 technical term만 영어를 유지한다. 실험도 같은 방식으로 `01_...`, `02_...` 순서를 유지한다.

## 현재 학습 가능 상태

- `00_foundations`: 5개 unit가 모두 채워져 있다.
- `01_ml`: 4개 stage와 최신 artifact가 정리돼 있다.
- `02_nlp_bridge -> 03_nlp`: bridge 2개 unit 뒤에 applied NLP 3개 unit가 바로 이어진다.
- `04_multimodal_bridge -> 05_multimodal`: bridge 1개 unit 뒤에 applied multimodal 3개 unit가 바로 이어진다.

즉 지금은 **00→05 전 구간을 하나씩 따라가며 공부할 수 있는 상태**다.

## 실험 산출물 규약

모든 실험은 최소한 아래 산출물을 남긴다.

- `config.yaml`: 하이퍼파라미터, 데이터 버전, seed
- `metrics.json`: 주요 지표
- `summary.md`: 한 번에 읽히는 실험 요약
- `figures/results/`: 학습 곡선, confusion matrix, retrieval 성능, caption 예시 등
- `figures/analysis/`: 에러 분석, slice 분석, feature importance, failure case panel 등
- `predictions/`: 샘플 예측 결과
- `model_card.md`: 승격할 가치가 있는 모델이면 작성

상세 규약은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md) 를 따른다.

## Git / 서버 / Hugging Face 운영 원칙

- 로컬에서는 문서 정리, 분석, 소규모 실험, 결과 선별을 담당한다.
- 서버에서는 실제 학습과 대량 로그/체크포인트 생성을 담당한다.
- `runs/` 는 기본적으로 Git에서 제외한다.
- 팀이 다시 볼 가치가 있는 결과만 `reports/` 와 `artifacts/promoted/` 로 승격한다.
- 작은 가중치는 `artifacts/promoted/` 아래에서 Git LFS로 관리할 수 있다.
- 큰 가중치는 Hugging Face Hub에 업로드하고, 링크와 커밋 정보를 [artifacts/MODEL_REGISTRY.md](artifacts/MODEL_REGISTRY.md) 에 기록한다.

Hugging Face 업로드와 Git LFS 관련 규칙은 루트의 `.gitignore`, `.gitattributes`, 그리고 [artifacts/README.md](artifacts/README.md) 에 정리했다.

## 참고 자료

- 전체 로드맵: [docs/00_program_map.md](docs/00_program_map.md)
- 순차 학습 가이드: [docs/02_study_guide.md](docs/02_study_guide.md)
- 실험 운영 규칙: [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md)
- 공식 참고 링크 모음: [docs/90_references.md](docs/90_references.md)
