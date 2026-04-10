# 01 Contrastive Alignment

## 왜 이 단위를 배우는가
`05_multimodal`로 가면 CLIP-style retrieval, image-text similarity, contrastive loss가 거의 바로 등장한다. 이 단위는 **이미지 벡터와 텍스트 벡터가 같은 의미일 때 대각선이 커지고, 다른 쌍일 때 비대각선이 작아져야 한다**는 가장 첫 감각을 한국어 tiny 예제로 먼저 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/alignment_heatmap.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 이미지/텍스트 toy embedding을 직접 정규화하고 cosine similarity와 contrastive loss를 계산한다.
2. `framework_lab.py`에서 PyTorch로 같은 정렬 문제를 CPU에서 shape-safe하게 계산하고 logits matrix를 확인한다.
3. `analysis.py`로 “왜 대각선이 정답 쌍인지”, “temperature가 어떤 sharpness를 만드는지”, “retrieval 전에 무엇을 봐야 하는지”를 한국어 문장으로 정리한다.

## 실행 결과 예시
```text
$ python 04_multimodal_bridge/01_contrastive_alignment/scratch_lab.py
{
  "pair_count": 3,
  "top1_alignment_accuracy": 1.0,
  "mean_positive_similarity": 0.996291,
  "mean_negative_similarity": 0.166819,
  "symmetric_contrastive_loss": 0.032464,
  "figure_path": "artifacts/scratch-manual/alignment_heatmap.svg"
}

$ python 04_multimodal_bridge/01_contrastive_alignment/framework_lab.py
{
  "device": "cpu",
  "logits_shape": [3, 3],
  "top1_alignment_accuracy": 1.0,
  "loss_i2t": 0.032493,
  "loss_t2i": 0.032435
}
```
실행 후에는 heatmap SVG와 metrics JSON이 모두 `artifacts/` 아래에 남아, contrastive alignment가 “정답 쌍의 대각선을 키우는 문제”라는 사실을 바로 눈으로 확인할 수 있다.

## 다음 단위와의 연결
이 감각이 있어야 `05_multimodal/01_image_text_retrieval`에서 zero-shot CLIP retrieval, hard negative, Recall@K를 숫자와 qualitative case로 읽을 수 있다. retrieval는 결국 “가장 가까운 텍스트/이미지를 찾는 문제”이므로, 여기서 만든 similarity matrix 읽기 습관이 바로 이어진다.
