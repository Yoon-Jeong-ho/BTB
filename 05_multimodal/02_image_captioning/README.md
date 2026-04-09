# 02 Image Captioning

## 왜 이 단위를 배우는가
`01_image_text_retrieval`에서 “가장 맞는 텍스트를 찾는 문제”를 봤다면, 이제는 이미지를 보고 **문장을 직접 생성하는 문제**로 넘어간다. 이 단위는 아주 작은 captioning 예제로 **decoder-style 생성, hallucination, 자동 지표와 사람 해석의 차이**를 한 번에 익히도록 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/caption_diagnostics.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 hand-crafted image feature와 간단한 언어 prior를 섞어 toy captioner를 만든다.
2. scratch 예제에서 unigram precision이 완전히 망가지지 않아도, 특정 장면에서 `dog` hallucination이 생길 수 있음을 확인한다.
3. `framework_lab.py`에서 CPU-safe PyTorch tiny decoder를 teacher forcing으로 학습하고, greedy decode 결과가 어떻게 안정되는지 본다.
4. `analysis.py`로 exact match, unigram precision, hallucination 사례를 함께 읽으며 “왜 captioning은 숫자 하나로 끝낼 수 없는가?”를 한국어 문장으로 정리한다.

## 실행 결과 예시
```text
$ python 05_multimodal/02_image_captioning/scratch_lab.py
{
  "sample_count": 4,
  "exact_match_rate": 0.75,
  "corpus_unigram_precision": 0.875,
  "hallucinated_content_tokens_total": 1,
  "figure_path": "artifacts/scratch-manual/caption_diagnostics.svg"
}

$ python 05_multimodal/02_image_captioning/framework_lab.py
{
  "device": "cpu",
  "epochs": 60,
  "token_accuracy": 1.0,
  "exact_match_rate": 1.0,
  "corpus_unigram_precision": 1.0
}
```
실행 후에는 caption diagnostics SVG, scratch/framework metrics JSON, 그리고 실행별 해석 리포트가 모두 `artifacts/` 아래에 남는다. 즉 captioning을 단순히 “문장을 뽑았다”가 아니라 **어떤 토큰이 hallucination 되었는지, decoder가 실제 추론에서 얼마나 안정적인지**로 읽게 된다.

## 이 단위에서 특히 볼 것
- unigram precision이 높아 보여도, 사람이 읽었을 때 틀린 subject token 하나가 caption 품질을 크게 망칠 수 있다.
- teacher forcing으로 loss가 내려가도 greedy decode에서는 여전히 반복/누락/환각이 생길 수 있다.
- `caption_diagnostics.svg`를 보면 샘플별 생성 길이와 hallucination count를 함께 볼 수 있어 qualitative inspection이 빨라진다.

## 다음 단위와의 연결
이 감각이 있어야 `03_visual_question_answering`에서 정답률 숫자만이 아니라 **왜 답이 틀렸는지, grounding이 어디서 끊겼는지**를 더 잘 설명할 수 있다. captioning에서 본 hallucination 감각은 VQA failure 해석에도 그대로 이어진다.
