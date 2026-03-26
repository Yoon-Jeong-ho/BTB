# 01 Experiment Playbook

## 1. Run ID 규약

권장 형식:

```text
YYYYMMDD-HHMMSS_<dataset>_<model>_s<seed>
```

예시:

```text
20260326-221500_nsmc_klue-roberta-base_s42
```

## 2. 모든 실험이 남겨야 하는 파일

```text
runs/<track>/<stage>/<run_id>/
├── config.yaml
├── metrics.json
├── summary.md
├── logs/
├── figures/
│   ├── results/
│   └── analysis/
├── predictions/
└── checkpoints/
```

필수 파일 의미:

- `config.yaml`: 데이터 경로, 모델, seed, optimizer, scheduler
- `metrics.json`: 최종 score와 best checkpoint 기준 score
- `summary.md`: 가설, 결과, 실패 원인, 다음 실험
- `figures/results/`: 최종 성능을 보여주는 그림
- `figures/analysis/`: 왜 그런 결과가 나왔는지 설명하는 그림

## 3. 결과 figure와 분석 figure 구분

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

## 4. Git에 올릴 것과 올리지 않을 것

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

## 5. 승격 기준

실험 결과를 `reports/` 로 옮길 때는 아래 질문에 모두 답할 수 있어야 한다.

1. 이전 실험과 무엇이 달라졌는가
2. metric이 정말 좋아졌는가
3. 그 차이를 figure로 설명 가능한가
4. 실패 사례를 최소 3개 이상 봤는가
5. 다음 실험이 무엇인지 명확한가

## 6. 작은 가중치 vs 큰 가중치

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

## 7. 로컬-서버 협업 흐름

1. 로컬에서 실험 설계와 config 정리
2. Git push
3. 서버에서 pull 후 학습 실행
4. `runs/` 에 원시 산출물 생성
5. 핵심 figure와 summary만 `reports/` 로 승격
6. 작은 모델은 `artifacts/promoted/`, 큰 모델은 HF Hub로 업로드
7. registry 갱신 후 Git push

## 8. 권장 로그 도구

- 기본 추천: MLflow
- 선택지: WandB, TensorBoard

여기서는 도구보다 산출물 규격 통일이 더 중요하다. 어떤 도구를 쓰더라도 최종적으로는 `metrics.json`, `summary.md`, `figures/` 를 남긴다.
