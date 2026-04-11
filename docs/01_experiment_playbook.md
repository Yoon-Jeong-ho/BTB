# 01 Experiment Playbook

## 1. 한 unit를 공부할 때 기본 순서

하나의 학습 단위는 아래 순서로 보는 것을 기본으로 한다.

1. `README.md`: 이 unit가 무엇을 배우는지와 대표 figure를 먼저 본다.
2. `THEORY.md`: 용어와 수식, 개념 연결을 읽는다.
3. `PREREQS.md`가 있으면 모르는 선행 개념이 있는지 빠르게 점검한다.
4. `scratch_lab.py`: 가장 작은 숫자 예제로 개념을 확인한다.
5. `framework_lab.py`: PyTorch / Transformers 등 실제 라이브러리 구현과 연결한다.
6. `analysis.md`: 실행 결과와 figure를 어떻게 해석해야 하는지 읽는다.
7. `reflection.md`: 스스로 설명할 수 있는지 점검한다.

## 2. 모든 실험/학습 단위가 남겨야 하는 파일

### 학습 단위 contract

아래 블록은 unit 기본 골격이다. 3절부터는 run/report artifact 승격 규약으로 읽는다.

```text
<unit>/
├── README.md
├── THEORY.md
├── PREREQS.md        # optional, 있는 unit만 읽는다
├── lesson.yaml
├── scratch_lab.py
├── framework_lab.py
├── analysis.md
├── reflection.md
└── artifacts/
```

- `lesson.yaml`: 목표, 선행 개념, 출력 계약, 분석 질문
- `analysis.md`: 결과 해설 문서
- `reflection.md`: 학습자 관점 회고
- runtime 관련 실습은 GPU/CPU 관측치를 함께 남긴다.
- runtime observations는 숫자만 적지 말고 원인 해석까지 붙인다.

## 3. 실행 예시

대표 실행 흐름은 아래와 같다.

```bash
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode scratch
python scripts/run_lesson.py --unit 00_foundations/01_tensor_shapes --mode framework
python scripts/build_lesson_report.py --unit 00_foundations/01_tensor_shapes
python scripts/check_curriculum_links.py
```

- `run_lesson.py`: unit의 `scratch_lab.py` 또는 `framework_lab.py`를 실행한다.
- `build_lesson_report.py`: unit의 요약 report 초안을 만든다.
- `check_curriculum_links.py`: README와 docs 링크가 깨지지 않았는지 점검한다.

## 4. Run ID 규약

권장 형식:

```text
YYYYMMDD-HHMMSS_<dataset>_<model>_s<seed>
```

예시:

```text
20260326-221500_nsmc_klue-roberta-base_s42
```

## 5. 결과 figure와 분석 figure 구분

### Results

- 학습 곡선
- confusion matrix
- ROC/PR curve
- Recall@K
- BLEU/CIDEr 표
- 정답률 요약 차트

### Analysis

- feature importance
- calibration
- slice metric
- error category bar chart
- 실패 사례 패널
- hallucination / boundary error / no-answer threshold 분석

## 6. Git에 올릴 것과 올리지 않을 것

### Git에 올릴 것

- `reports/` 아래의 핵심 요약
- 대표 figure
- `metrics.json`
- 작은 모델 가중치와 model card
- Hugging Face 링크가 포함된 registry

### Git에 올리지 않을 것

- 원시 데이터
- 대량 로그
- sweep 전체 checkpoint
- 재생성 가능한 cache

## 7. 승격 기준

실험 결과를 `reports/` 로 옮길 때는 아래 질문에 모두 답할 수 있어야 한다.

1. 이전 실험과 무엇이 달라졌는가
2. metric이 정말 좋아졌는가
3. 그 차이를 figure로 설명 가능한가
4. 실패 사례를 최소 3개 이상 봤는가
5. 다음 실험이 무엇인지 명확한가

## 8. 작은 가중치 vs 큰 가중치

### 작은 가중치

- `artifacts/promoted/` 에 저장
- Git LFS 사용
- 대응 `model_card.md` 필수

### 큰 가중치

- Hugging Face Hub 사용
- 이 저장소에는
  - HF repo 링크
  - 대응 Git commit
  - 데이터셋
  - 핵심 metrics
  만 남긴다

## 9. 로컬-서버 협업 흐름

1. 로컬에서 실험 설계와 config 정리
2. Git push
3. 서버에서 pull 후 학습 실행
4. `runs/` 에 원시 산출물 생성
5. 핵심 figure와 summary만 `reports/` 로 승격
6. 작은 모델은 `artifacts/promoted/`, 큰 모델은 HF Hub로 업로드
7. registry 갱신 후 Git push

## 10. 권장 로그 도구

- 기본 추천: MLflow
- 선택지: WandB, TensorBoard

여기서는 도구보다 산출물 규격 통일이 더 중요하다. 어떤 도구를 쓰더라도 최종적으로는 `metrics.json`, `summary.md`, `figures/` 를 남긴다.
