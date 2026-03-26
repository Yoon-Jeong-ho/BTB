# 03 Machine Reading Comprehension

## 목표

`질문-문맥 정렬`, `span extraction`, `불답 처리` 를 실험적으로 익힌다.

## 추천 데이터셋

- `KLUE-MRC`: 한국어 독해
- `SQuAD 2.0`: answerable / unanswerable 구분이 포함된 표준 독해 벤치마크

## 실습 파이프라인

1. context/question/answer length 통계 확인
2. baseline으로 retrieval + heuristic 또는 작은 QA head
3. pretrained QA model finetuning
4. EM, F1, no-answer threshold 분석
5. answerable / unanswerable 성능 분리 분석
6. 긴 문맥, 애매한 질문, 여러 후보 span 사례 분석

## 결과로 남길 figure

- question/context length histogram
- train/valid curve
- EM/F1 comparison chart
- no-answer threshold curve

## 분석으로 남길 figure

- answerable vs unanswerable breakdown
- top failure span table
- long-context slice metric

## 승격 기준

- span을 틀린 이유가 문맥 부족인지, 질문 해석 실패인지, threshold 문제인지 구분된다.
