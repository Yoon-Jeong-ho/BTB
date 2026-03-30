# reflection

## 내가 직접 설명해 보기
- similarity matrix에서 대각선이 정답 쌍이라는 말을, 내가 만든 예시 하나로 다시 설명해 보자.
- mean positive similarity와 hardest negative similarity 차이가 작아지면 retrieval에서 어떤 실수가 먼저 늘어날까?
- image→text와 text→image 중 어느 방향이 더 어려울 수 있는지, 실제 서비스 예시를 떠올려 적어 보자.

## 실행 후 체크
- `alignment_heatmap.svg`에서 가장 진한 칸이 모두 대각선이었는가?
- scratch와 framework 결과의 loss가 같은 직관을 주었는가?
- 이 단위의 감각이 `05_multimodal/01_image_text_retrieval`의 Recall@K 해석과 어떻게 이어지는가?
