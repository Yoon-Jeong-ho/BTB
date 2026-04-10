# 01 Image-Text Retrieval

## 왜 이 단위를 배우는가
`04_multimodal_bridge/01_contrastive_alignment`에서 “대각선이 살아야 한다”는 감각을 만들었다면, 이제는 그 감각을 **실제 retrieval 지표**로 읽어야 한다. 이 단위는 아주 작은 이미지-텍스트 예제에서 **Recall@K, 양방향 검색(image→text / text→image), hard negative**를 숫자와 사례로 동시에 확인하게 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/retrieval_heatmap.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 hand-crafted image/text embedding으로 similarity matrix와 Recall@K를 직접 계산한다.
2. scratch 예제에서 **image→text는 맞지만 text→image 한 방향은 흔들리는** asymmetric retrieval case를 확인한다.
3. `framework_lab.py`에서 CPU-safe PyTorch dual encoder를 아주 작은 raw feature 입력에 맞춰 학습해, 양방향 retrieval이 어떻게 안정되는지 본다.
4. `analysis.py`로 scratch와 framework를 비교하며 “왜 Recall@1만 보면 부족한지”, “hard negative를 어떻게 읽어야 하는지”를 한국어 문장으로 정리한다.

## 실행 결과 예시
```text
$ python 05_multimodal/01_image_text_retrieval/scratch_lab.py
{
  "pair_count": 4,
  "image_to_text_recall_at_1": 1.0,
  "text_to_image_recall_at_1": 0.75,
  "text_to_image_recall_at_2": 1.0,
  "hardest_negative_pair": "주황빛 호수 카약 ↔ 숲속 강아지 산책",
  "figure_path": "artifacts/scratch-manual/retrieval_heatmap.svg"
}

$ python 05_multimodal/01_image_text_retrieval/framework_lab.py
{
  "device": "cpu",
  "epochs": 200,
  "image_to_text_recall_at_1": 1.0,
  "text_to_image_recall_at_1": 1.0,
  "symmetric_loss": 0.0069
}
```
실행 후에는 heatmap SVG, scratch/framework metrics JSON, 그리고 실행별 해석 리포트가 모두 `artifacts/` 아래에 남는다. 즉 retrieval를 단순한 “유사도 계산”이 아니라 **검색 방향과 ranking 지표를 읽는 문제**로 받아들이는 연습이 된다.

## 이 단위에서 특히 볼 것
- 같은 similarity matrix라도 **행 기준(image→text)** 과 **열 기준(text→image)** 으로 읽으면 다른 failure가 드러날 수 있다.
- Recall@1이 완벽하지 않아도 Recall@2가 높다면, 후보군 안에는 정답이 들어왔지만 top ranking이 아직 불안정하다는 뜻이다.
- hard negative가 높게 남으면 qualitative retrieval panel에서 실패 사례를 먼저 확인해야 한다.

## 다음 단위와의 연결
이 감각이 있어야 `02_image_captioning`에서 생성 결과를 retrieval 관점과 비교해 읽을 수 있고, `03_visual_question_answering`에서는 단순 matching이 아닌 grounded reasoning failure를 더 분명하게 구분할 수 있다.
