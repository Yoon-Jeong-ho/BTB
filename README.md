# BTB

## NLP 바보에서 박사

이 저장소는 `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 순서로 올라가면서, 이론을 읽고 끝내지 않고 반드시 작은 실험으로 검증하는 학습 저장소다.

핵심 철학은 세 가지다.

1. 쉬운 베이스라인부터 시작한다.
2. 모든 실험은 `로그`, `결과 figure`, `분석 figure`, `실패 사례`를 남긴다.
3. Git에는 이해와 재현에 필요한 산출물을 남기고, 큰 가중치는 Hugging Face Hub로 분리한다.

## 학습 순서

1. [00_foundations](00_foundations/README.md): 텐서, gradient, attention, GPU/runtime 같은 공통 기초를 먼저 다진다.
2. [01_ml](01_ml/README.md): 표형 데이터, metric, error analysis, experiment discipline을 익힌다.
3. [02_nlp_bridge](02_nlp_bridge/README.md): tokenization, embedding, sequence modeling, transformer 감각을 연결한다.
4. [03_nlp](03_nlp/README.md): 본격 NLP 실습 트랙으로 이어질 자리를 미리 고정한다.
5. [04_multimodal_bridge](04_multimodal_bridge/README.md): alignment, retrieval vs generation, cross-attention을 멀티모달 전에 다진다.
6. [05_multimodal](05_multimodal/README.md): 멀티모달 실습 트랙으로 이어질 자리를 미리 고정한다.

현재 실제 콘텐츠는 단계적으로 재배치 중이지만, 루트 탐색 경험은 위 인덱스 사다리를 기준으로 유지한다.

전체 프로그램 개요는 [docs/00_program_map.md](docs/00_program_map.md), 실험 운영 규칙은 [docs/01_experiment_playbook.md](docs/01_experiment_playbook.md) 에 정리했다.

## 저장소 구조

```text
BTB/
├── 00_foundations/             # 공통 기초 트랙
├── 00_shared/                  # 공통 규약, 템플릿
├── 01_ml/                      # 기초 ML 트랙
├── 02_nlp_bridge/              # ML -> NLP 브리지
├── 03_nlp/                     # NLP 트랙(재배치 대상 인덱스)
├── 04_multimodal_bridge/       # NLP -> 멀티모달 브리지
├── 05_multimodal/              # 멀티모달 트랙(재배치 대상 인덱스)
├── data/                       # raw/interim/processed/external 설명용 구조
├── runs/                       # 서버/로컬의 비정제 실행 산출물(기본 ignore)
├── reports/                    # Git에 남길 승격된 실험 결과
├── artifacts/                  # 모델 가중치/레지스트리 규칙
├── docs/                       # 프로그램 개요, 운영 문서, 참고 자료
└── scripts/                    # 향후 학습/평가 스크립트 인터페이스 규약
```

최상위 폴더와 각 단계 폴더에 인덱스를 붙여 정렬이 무너지지 않게 했다. 문서는 한글 우선으로 쓰고, 코드/파일명/핵심 technical term만 영어를 유지한다. 실험도 같은 방식으로 `01_...`, `02_...` 순서를 유지한다.

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

## 시작 순서

1. [docs/00_program_map.md](docs/00_program_map.md) 를 읽고 전체 로드맵을 본다.
2. [00_foundations/README.md](00_foundations/README.md) 에서 공통 기초를 확인한 뒤 [01_ml/README.md](01_ml/README.md) 의 `01_tabular_classification` 부터 시작한다.
3. 실험을 돌릴 때는 [00_shared/templates/run_summary_template.md](00_shared/templates/run_summary_template.md) 형식으로 요약을 남긴다.
4. 결과 중 다시 볼 가치가 있는 것만 [reports/README.md](reports/README.md) 규칙에 맞게 승격한다.
5. 가중치를 공유할 때는 [artifacts/MODEL_REGISTRY.md](artifacts/MODEL_REGISTRY.md) 를 먼저 갱신한다.

## 참고 자료

사용한 공식 사이트, 벤치마크, GitHub, 문서는 [docs/90_references.md](docs/90_references.md) 에 모아 두었다.
