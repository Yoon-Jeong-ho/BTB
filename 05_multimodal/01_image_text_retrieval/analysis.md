# 01 Image-Text Retrieval 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 image-text retrieval를 해석하는 안정적인 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- retrieval의 첫 질문은 “유사도 숫자가 높은가?”가 아니라, **정답이 ranking 몇 위에 들어왔는가**다.
- image→text 와 text→image 는 같은 matrix를 보더라도 query 방향이 달라 서로 다른 failure를 만든다. 따라서 두 방향을 따로 기록해야 한다.
- Recall@1이 흔들리더라도 Recall@2 또는 Recall@5가 유지된다면, 정답 후보는 살아 있지만 top-1 calibration이 약하다는 뜻이다.
- hard negative는 retrieval 모델이 무엇을 헷갈리는지 보여 주는 가장 빠른 qualitative 단서다. 숫자와 사례를 함께 읽어야 한다.
- tiny PyTorch dual encoder는 “두 modality 입력이 원래 같은 공간이 아니다”라는 사실을 작게 재현한다. projection을 학습해 같은 좌표계로 옮기는 과정 자체가 retrieval의 핵심이다.

## 확인 질문
- 이번 unit에서 가장 높은 hard negative는 어떤 쌍이었고, 왜 헷갈렸는가?
- image→text 와 text→image 중 어느 방향이 더 어려웠으며, 그 이유를 similarity matrix로 설명할 수 있는가?
- framework에서 Recall@1이 개선되었다면, 그것이 shared embedding space 관점에서 무엇을 의미하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): Recall@K, hard negative, shared embedding space, symmetric contrastive loss 개념을 다시 확인한다.
