# 01. 표형 분류 한눈 요약

- stage 가이드: [README](../../README.md)
- 이론 문서: [THEORY](../../THEORY.md)
- 상세 리포트: [artifact README](README.md)

## 핵심 결론

`random_forest` 가 AUPRC 기준으로 가장 좋았다.
하지만 이 실험의 진짜 포인트는 “최고 점수 모델”보다 **왜 accuracy 와 AUPRC 의 결론이 다를 수 있는지**를 이해하는 데 있다.

## 꼭 기억할 메트릭

- `AUPRC=0.7834`: 희소 positive class 탐지의 핵심 지표
- `AUROC=0.9105`: 전체 ranking 품질
- `F1=0.6971`: threshold 기준 precision/recall 균형
- `Accuracy=0.8354`: 불균형 때문에 과대평가될 수 있음

## 고확신 오답에서 보인 패턴

- 성별: {'Male': 29, 'Female': 1}
- 학력 상위 3개: {'Bachelors': 11, 'Masters': 9, 'Doctorate': 5}
- 평균 나이 / 근무시간: 44.9세 / 47.9시간

이 패턴은 모델이 특정 사회경제적 archetype 을 과하게 믿는다는 뜻이다.

## 먼저 볼 figure

1. ![](figures/results/pr_curve.svg)
2. ![](figures/results/confusion_matrix.svg)
3. ![](figures/analysis/error_slice_by_sex.svg)
4. ![](figures/analysis/failure_examples.svg)

## 한 문장 해석

Stage 1 은 “분류를 한다”가 아니라 **분류 score 를 해석하고, 메트릭을 구분하고, high-confidence mistake 를 읽는 법을 배우는 단계**다.
