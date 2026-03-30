# 01 Contrastive Alignment 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 조금 달라져도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- contrastive alignment는 이미지와 텍스트를 같은 공간에 놓고, 정답 쌍이 대각선에서 가장 높은 similarity를 받게 만드는 문제다.
- similarity matrix를 읽을 때는 절대값 하나보다 **대각선과 비대각선의 간격** 이 더 중요하다.
- retrieval 직전에는 top-1 accuracy뿐 아니라 hardest negative가 얼마나 높게 남는지도 봐야 한다.
- image→text와 text→image를 둘 다 보는 이유는, 한 방향만 쉬우면 실제 검색 시스템에서 편향된 정렬이 남을 수 있기 때문이다.

## 확인 질문
- 왜 정답 쌍은 similarity matrix 대각선에 놓는가?
- 양의 쌍 평균 유사도와 음의 쌍 평균 유사도 차이가 작아지면 retrieval는 어떻게 흔들리는가?
- temperature가 작아질수록 logits와 gradient는 어떤 쪽으로 더 날카로워지는가?
- 한 이미지에 여러 caption이 가능한 실제 데이터셋에서는 이 toy setting을 어떻게 확장해야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): contrastive alignment, joint embedding space, temperature, bidirectional retrieval를 다시 확인한다.
