# 01 Image-Text Retrieval 선행 개념

## 꼭 알고 오면 좋은 것
- cosine similarity 또는 dot product가 임베딩 유사도를 어떻게 표현하는지
- 행렬의 행/열을 query 방향으로 읽는 기본 습관
- contrastive alignment가 왜 shared embedding space를 만드는지

## 빠른 자기 점검
- similarity matrix가 주어졌을 때 각 행의 top-1 텍스트를 읽을 수 있는가?
- `Recall@1` 과 `Recall@5` 의 차이를 한 문장으로 설명할 수 있는가?
- image→text 와 text→image 가 왜 서로 다른 failure를 만들 수 있는지 말할 수 있는가?
