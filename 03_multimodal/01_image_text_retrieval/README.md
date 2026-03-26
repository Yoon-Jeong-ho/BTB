# 01 Image Text Retrieval

## 목표

이미지와 텍스트를 같은 임베딩 공간에 정렬시키고 `Recall@K` 와 정성 사례로 평가하는 법을 익힌다.

## 추천 데이터셋

- `MS COCO` retrieval split
- 계산량이 크면 COCO subset부터 시작

## 실습 파이프라인

1. image-text pair 정합성 확인
2. pretrained CLIP 임베딩으로 zero-shot retrieval baseline
3. hard negative mining 여부 기록
4. VisionTextDualEncoder 또는 CLIP-style contrastive finetuning
5. Recall@1/5/10 계산
6. retrieval success / failure qualitative panel 작성

## 결과로 남길 figure

- Recall@K bar chart
- training loss curve
- text-to-image / image-to-text retrieval grid

## 분석으로 남길 figure

- embedding projection plot
- hard negative failure cases
- caption length / object count별 성능 slice

## 승격 기준

- 숫자와 qualitative panel이 함께 개선된다.
- 어떤 종류의 이미지-텍스트 쌍에서 정렬이 무너지는지 설명 가능하다.
