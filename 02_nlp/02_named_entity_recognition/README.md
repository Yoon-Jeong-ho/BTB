# 02 Named Entity Recognition

## 목표

토큰 단위 예측에서 `label alignment`, `boundary error`, `entity-level F1` 을 익힌다.

## 추천 데이터셋

- `KLUE-NER`: 한국어 NER
- `CoNLL-2003`: 영어 NER 표준 벤치마크

## 실습 파이프라인

1. BIO tagging 규약 확인
2. tokenizer와 word-piece alignment 점검
3. baseline으로 CRF 계열 또는 간단한 sequence model
4. transformer token classification finetuning
5. entity-level F1와 label별 F1 비교
6. 경계 오류와 label confusion 분석

## 결과로 남길 figure

- entity label distribution
- sequence length histogram
- per-label F1 bar chart

## 분석으로 남길 figure

- boundary error examples
- label confusion summary
- 긴 문장/짧은 문장 slice metric

## 승격 기준

- 토큰 단위 accuracy보다 entity-level F1에 집중한다.
- 어떤 엔티티 타입에서 boundary가 무너지는지 설명 가능하다.
