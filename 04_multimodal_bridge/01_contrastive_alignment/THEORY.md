# 01 Contrastive Alignment 이론 노트

## 핵심 개념
- **contrastive alignment**는 같은 의미를 가리키는 이미지-텍스트 쌍의 임베딩을 가깝게, 다른 쌍은 멀게 만드는 학습 원리다.
- **joint embedding space**는 이미지 encoder와 텍스트 encoder 출력이 같은 좌표계에서 비교될 수 있도록 맞춘 공간이다.
- **similarity matrix**는 `(image batch, text batch)` 모든 쌍의 유사도를 표로 펼친 것이다. 정답 쌍은 보통 대각선에 놓는다.
- **temperature**는 logits를 얼마나 날카롭게 볼지 정하는 스케일이다. temperature가 작을수록 높은 similarity를 더 강하게 구분한다.
- **bidirectional retrieval**는 image→text 뿐 아니라 text→image 방향도 함께 잘 되어야 한다는 뜻이다.

## 수식 / 직관
- 정규화된 임베딩이 있으면 cosine similarity는 거의 dot product로 읽을 수 있다.
- `S = normalize(I) @ normalize(T)^T` 이면 `S[i, j]`는 `i`번째 이미지와 `j`번째 텍스트의 유사도다.
- contrastive logits는 보통 `S / temperature` 로 만든다.
- 배치 정답이 `[0, 1, 2, ...]` 이라면, cross entropy는 각 행과 각 열에서 대각선 값이 가장 크도록 압박한다.

## 이 단위에서 꼭 볼 것
- 대각선 평균 similarity가 비대각선 평균보다 충분히 큰가?
- top-1 정렬 정확도가 1.0이더라도, hard negative 비대각선이 얼마나 높게 남는가?
- image→text와 text→image loss가 둘 다 낮은가, 아니면 한 방향만 쉬운가?

## Common Confusion
- contrastive alignment를 “문장을 생성하는 문제”로 오해하는 실수
- cosine similarity가 높으면 언제나 semantic understanding이 완전하다고 과대해석하는 실수
- temperature를 단순 scaling으로만 보고 gradient sharpness 변화는 놓치는 실수
- 한 이미지에 여러 정답 caption이 있을 수 있는데, 배치에서 하나만 정답이라고 너무 단순화하는 실수

## 실행에서 확인할 포인트
- `artifacts/scratch-manual/alignment_heatmap.svg`에서 대각선 칸이 가장 진하게 보이는지 확인한다.
- scratch와 framework 결과 모두에서 `logits_shape == [3, 3]` 인지 확인한다.
- 양의 쌍 평균 유사도와 음의 쌍 평균 유사도 차이가 충분한지 본다.
- symmetric loss를 image→text / text→image 두 방향으로 나눠 읽는다.

## 실행 결과 예시
```text
scratch metrics
- top1_alignment_accuracy: 1.0
- mean_positive_similarity: 0.996291
- mean_negative_similarity: 0.166819
- symmetric_contrastive_loss: 0.032464

framework metrics
- device: cpu
- logits_shape: [3, 3]
- loss_i2t: 0.032493
- loss_t2i: 0.032435
```
이 숫자는 “같은 장면을 가리키는 이미지/텍스트 임베딩은 대각선에서 가장 큰 점수를 받아야 한다”는 contrastive retrieval의 핵심 직관을 아주 작은 배치에서 보여준다.
