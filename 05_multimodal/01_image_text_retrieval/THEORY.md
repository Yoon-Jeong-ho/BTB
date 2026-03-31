# 01 Image-Text Retrieval 이론 노트

## 핵심 개념
- **image-text retrieval**는 이미지에서 맞는 텍스트를 찾거나, 텍스트에서 맞는 이미지를 찾는 ranking 문제다.
- **bidirectional retrieval**는 image→text 와 text→image 두 방향 모두를 separately 평가해야 한다는 뜻이다.
- **Recall@K** 는 정답이 상위 `K`개 후보 안에 들어오면 성공으로 본다. `Recall@1`은 가장 엄격하고, `Recall@5`는 더 관대한 지표다.
- **hard negative**는 정답은 아니지만 점수가 높아 검색을 헷갈리게 만드는 음성 쌍이다.
- **shared embedding space**는 이미지 encoder와 텍스트 encoder 출력이 같은 좌표계에서 dot product 또는 cosine similarity로 비교되는 공간이다.

## 수식 / 직관
- 정규화된 임베딩을 `I`, `T` 라고 하면 similarity matrix는 `S = normalize(I) @ normalize(T)^T` 로 쓸 수 있다.
- `S[i, j]` 는 `i`번째 이미지와 `j`번째 텍스트가 얼마나 가깝게 정렬됐는지 보여 준다.
- retrieval에서는 단순히 대각선이 큰지만 보지 않고, **정답이 ranking 몇 위인가**를 함께 본다.
- symmetric contrastive loss는 image→text 와 text→image cross entropy를 평균 내어, 한 방향만 잘 되는 편향을 줄인다.

## 이 단위에서 꼭 볼 것
- scratch에서 `image_to_text_recall_at_1` 과 `text_to_image_recall_at_1` 이 왜 다르게 나오는가?
- `Recall@1` 이 흔들려도 `Recall@2` 가 유지되면 “정답 후보는 들어왔지만 ranking calibration이 약하다”는 해석이 가능한가?
- framework 학습 후에는 hard negative가 얼마나 줄고 symmetric loss가 얼마나 낮아졌는가?
- qualitative retrieval row를 읽을 때, 어떤 설명이 어떤 이미지와 헷갈렸는지를 말로 설명할 수 있는가?

## Common Confusion
- retrieval를 단순 분류 정확도처럼 하나의 숫자로만 읽는 실수
- image→text 성능만 보고 text→image 방향을 생략하는 실수
- Recall@K가 높다고 해서 hard negative가 사라졌다고 착각하는 실수
- similarity matrix 대각선만 보고 ranking 순위 자체는 확인하지 않는 실수

## PyTorch tiny demo에서 보는 구조
- 이미지 raw feature와 텍스트 bag-of-concepts 입력은 원래 같은 공간이 아니다.
- 그래서 `framework_lab.py`에서는 **작은 두 개의 projection encoder**를 학습해 shared embedding space를 만든다.
- 이 tiny demo는 대규모 CLIP 학습이 아니라도, retrieval가 결국 “두 modality를 같은 좌표계로 옮겨 순위를 매기는 문제”라는 점을 분명하게 보여 준다.

## 실행 결과 예시
```text
scratch metrics
- image_to_text_recall_at_1: 1.0
- text_to_image_recall_at_1: 0.75
- text_to_image_recall_at_2: 1.0
- hardest_negative_pair: 주황빛 호수 카약 ↔ 숲속 강아지 산책

framework metrics
- device: cpu
- logits_shape: [4, 4]
- image_to_text_recall_at_1: 1.0
- text_to_image_recall_at_1: 1.0
- symmetric_loss: 0.0069
```
이 숫자는 “대각선이 큰가?”를 넘어, **정답이 ranking 몇 위에 들어왔는지와 양방향 검색이 같이 안정적인지**를 봐야 retrieval를 제대로 읽을 수 있음을 보여 준다.
