# 01. 표형 분류 실행 요약

- 과제: Adult Census Income 이진 분류
- 최고 모델: `random_forest`
- 핵심 지표: AUPRC=0.7834, AUROC=0.9105, F1=0.6971, Accuracy=0.8354

## 모델 비교

| 모델 | AUPRC | AUROC | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| random_forest | 0.7834 | 0.9105 | 0.6971 | 0.8354 |
| logistic_regression | 0.7657 | 0.9044 | 0.6724 | 0.8055 |
| gpu_mlp | 0.7569 | 0.9021 | 0.6851 | 0.8510 |
| dummy_prior | 0.2407 | 0.5000 | 0.0000 | 0.7593 |

## 파일 둘러보기

- 이론 노트: [../../THEORY.md](../../THEORY.md)
- stage 가이드: [../../README.md](../../README.md)
- 결과 figure: `figures/results/`
- 분석 figure: `figures/analysis/`
- 고확신 오답: `predictions/high_confidence_errors.csv`
