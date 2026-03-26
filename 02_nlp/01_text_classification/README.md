# 01 Text Classification

## 목표

텍스트 분류에서 `간단한 sparse baseline` 과 `pretrained LM finetuning` 의 차이를 분명히 체감한다.

## 추천 데이터셋

- `NSMC`: 한국어 감성 분류 입문
- `IMDb`: 영어 감성 분류 표준
- `KLUE-TC (YNAT)`: 뉴스 토픽 분류 확장

## 실습 파이프라인

1. 데이터 카드 작성과 라이선스 확인
2. 길이 분포, 클래스 분포 분석
3. baseline으로 `TF-IDF + LogisticRegression` 또는 linear SVM
4. pretrained tokenizer와 encoder 로딩
5. transformer finetuning
6. accuracy, macro F1, calibration 비교
7. 오분류 문장과 borderline sample 분석

## 결과로 남길 figure

- text length histogram
- class distribution
- learning curve
- confusion matrix
- calibration plot

## 분석으로 남길 figure

- wrong prediction table
- confidence histogram
- class pair confusion summary

## 승격 기준

- sparse baseline 대비 transformer의 이득이 설명 가능하다.
- 길이/도메인/표현 방식에 따른 약점이 드러난다.
