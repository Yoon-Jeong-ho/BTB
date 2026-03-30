# 01 Contrastive Alignment 실행 관측

## 관측 결과
- pair count: `3`
- similarity matrix shape: `[3, 3]`
- logits shape: `[3, 3]`
- temperature: `0.2`
- top-1 alignment accuracy: `1.0`
- mean positive similarity: `0.996291`
- mean negative similarity: `0.166819`
- hardest negative similarity: `0.247587`
- positive-hard-negative gap: `0.748704`
- image→text loss: `0.032493`
- text→image loss: `0.032435`

## 한국어 해석
- 정답 쌍 평균 유사도 `0.996291` 가 음의 쌍 평균 `0.166819` 보다 충분히 높아, 이 tiny 배치에서는 대각선이 분명하게 살아 있다.
- hardest negative가 `0.247587` 로 남아 있다는 것은 완벽한 분리가 아니라, retrieval 관점에서는 여전히 헷갈릴 수 있는 비슷한 설명이 존재한다는 뜻이다.
- positive-hard-negative gap이 `0.748704` 이므로, 지금은 정답 쌍이 이긴다. 하지만 실제 데이터셋에서 이 간격이 줄면 Recall@K가 먼저 흔들리기 쉽다.
- framework 실행에서 `logits_shape = [3, 3]` 와 `device = cpu` 를 확인했으므로, CPU에서도 contrastive/logit-similarity 계산 흐름을 안전하게 재현했다.
- image→text loss와 text→image loss를 함께 보는 이유는 검색 방향 둘 다 안정적으로 정렬되어야 하기 때문이다.

## 정렬 상태 메모
- 이번 실행에서는 세 쌍 모두 top-1에서 자기 짝을 가장 가깝게 찾았다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
