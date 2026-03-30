# BTB Curriculum Redesign PR Draft

## Suggested PR title

BTB를 foundation-first 한글 학습 사다리로 재구성

## Why

이 PR은 BTB를 단순 실험 저장소에서 **읽으면서 따라갈 수 있는 한글 중심 학습 레포**로 바꾸기 위한 기반 작업이다.

핵심 목적은 세 가지다.

1. `00_foundations -> 01_ml -> 02_nlp_bridge -> 03_nlp -> 04_multimodal_bridge -> 05_multimodal` 흐름을 루트 구조에서 바로 보이게 만들기
2. 공통 학습 단위 contract를 만들고 gold-standard unit로 실제 품질 기준을 세우기
3. 최소 자동화(`lesson.yaml`, lesson runner, report scaffold, markdown link checker)를 붙여 이후 단위 확장을 쉽게 만들기

## What changed

### 1) 루트 구조와 문서 진입점 정리
- `.gitignore`에 `.superpowers/` 추가
- `README.md`를 새 인덱싱 구조 기준으로 재작성
- `docs/00_program_map.md`를 foundation/bridge 구조 기준으로 재작성
- `02_nlp/` → `03_nlp/`, `03_multimodal/` → `05_multimodal/` 재인덱싱
- `00_foundations/`, `02_nlp_bridge/`, `04_multimodal_bridge/` 도입

### 2) 공통 unit contract / template 추가
- `00_shared/README.md`를 문서형 unit contract 중심으로 재작성
- foundation용 템플릿 4종 추가
- `docs/01_experiment_playbook.md`에 `lesson.yaml`, `analysis.md`, `reflection.md`, runtime observation 규칙 추가

### 3) gold-standard foundations unit 2개 추가
- `00_foundations/01_tensor_shapes`
- `00_foundations/05_gpu_memory_runtime`

두 unit 모두 아래를 포함한다.
- `README.md`
- `THEORY.md`
- `PREREQS.md`
- `lesson.yaml`
- `scratch_lab.py`
- `framework_lab.py`
- `analysis.py`
- `analysis.md`
- `reflection.md`
- `artifacts/.gitkeep`

### 4) 자동화 스캐폴드 추가
- `scripts/_lesson_metadata.py`
- `scripts/run_lesson.py`
- `scripts/build_lesson_report.py`
- `scripts/check_curriculum_links.py`

추가로 다음을 보장하도록 강화했다.
- required outputs가 없으면 report builder가 **명시적으로 실패**함
- generated analysis 문서가 `THEORY.md`로 **역링크(backlink)** 를 가짐
- empty-alt 이미지 링크도 link checker가 검증함

### 5) 첫 concrete NLP bridge unit 추가
- `02_nlp_bridge/01_tokenization_and_embeddings`

이 unit은 다음을 다룬다.
- toy vocab 기반 tokenization / subword-ish splitting
- token id mapping
- embedding lookup shape 변화
- padding mask shape 감각
- 한국어 analysis + theory backlink

## Main files to review

### Topology / docs
- `README.md`
- `docs/00_program_map.md`
- `00_shared/README.md`
- `docs/01_experiment_playbook.md`

### Foundations units
- `00_foundations/01_tensor_shapes/*`
- `00_foundations/05_gpu_memory_runtime/*`

### NLP bridge unit
- `02_nlp_bridge/README.md`
- `02_nlp_bridge/01_tokenization_and_embeddings/*`

### Automation
- `scripts/_lesson_metadata.py`
- `scripts/run_lesson.py`
- `scripts/build_lesson_report.py`
- `scripts/check_curriculum_links.py`

### Tests
- `tests/test_curriculum_topology.py`
- `tests/test_reindexed_tracks.py`
- `tests/test_shared_templates.py`
- `tests/test_foundations_unit_contract.py`
- `tests/test_lesson_runner_contract.py`
- `tests/test_curriculum_links.py`
- `tests/test_nlp_bridge_unit_contract.py`

## Validation

실행한 최종 검증:

```bash
python -m unittest \
  tests/test_curriculum_topology.py \
  tests/test_reindexed_tracks.py \
  tests/test_shared_templates.py \
  tests/test_foundations_unit_contract.py \
  tests/test_lesson_runner_contract.py \
  tests/test_curriculum_links.py \
  tests/test_nlp_bridge_unit_contract.py \
  tests/test_01_ml_report_contract.py -v

python scripts/check_curriculum_links.py
```

결과:
- 전체 테스트 PASS
- markdown link checker PASS
- 작업 트리 clean

## Commit guide

이 변경은 아래 흐름으로 나뉘어 있다.
- `ecb6ec2` top-level ladder
- `d0fb129` reindex tracks
- `3f9edeb` shared contract/templates
- `3eb8c12` tensor shapes unit
- `13906e1` GPU/runtime unit
- `8a9fe15` automation scaffold
- `835d65f` automation contract hardening
- `e2700e6` first NLP bridge unit

## Reviewer checklist

- [ ] 루트 README만 읽어도 새 학습 흐름이 이해되는가?
- [ ] foundation/bridge/applied 구조가 번호 체계로 자연스럽게 읽히는가?
- [ ] foundation unit 2개가 품질 기준으로 충분히 읽을 만한가?
- [ ] NLP bridge 첫 unit가 03_nlp로 넘어가기 전 개념 다리 역할을 하는가?
- [ ] automation scripts가 과하게 무겁지 않고, 현재 범위에 맞게 단순한가?
- [ ] generated docs/report가 evidence-first 원칙을 지키는가?

## Remaining follow-ups

이 PR 이후 바로 이어갈 만한 작업:
1. `02_nlp_bridge`의 다음 unit (`attention` / `transformer block`) 추가
2. `04_multimodal_bridge` 첫 concrete unit 추가
3. 더 많은 unit가 생기면 `lesson.yaml` 스키마를 lint/validate하는 도구 추가
