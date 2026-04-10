# 03 Visual Question Answering

## 왜 이 단위를 배우는가
`02_image_captioning`에서 이미지 내용을 문장으로 풀어냈다면, 이제는 그 장면을 바탕으로 **질문에 맞는 짧은 답을 선택하는 문제**로 넘어간다. 이 단위는 아주 작은 VQA-style 예제로 **answer type(yes/no, color, count), shortcut bias, grounded reasoning failure**를 한 번에 읽게 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch 시각화 `artifacts/scratch-manual/vqa_answer_type_accuracy.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 hand-crafted image feature와 질문 메타데이터를 사용해 toy VQA 규칙기를 만든다.
2. scratch 예제에서 **yes/no와 color는 잘 맞지만 count는 prior 때문에 흔들리는** answer-type behavior를 확인한다.
3. `framework_lab.py`에서 CPU-safe PyTorch tiny multimodal classifier를 학습해, 이미지 특징과 질문 토큰을 함께 읽는 작은 VQA 파이프라인을 재현한다.
4. `analysis.py`로 scratch와 framework를 비교하며 “왜 VQA에서 overall accuracy만 보면 부족한가?”, “count 질문이 왜 특히 어려운가?”를 한국어 문장으로 정리한다.

## 실행 결과 예시
```text
$ python 09_multimodal/03_visual_question_answering/scratch_lab.py
{
  "sample_count": 6,
  "overall_accuracy": 0.833333,
  "answer_type_accuracy": {
    "yes/no": 1.0,
    "color": 1.0,
    "count": 0.5
  },
  "figure_path": "artifacts/scratch-manual/vqa_answer_type_accuracy.svg"
}

$ python 09_multimodal/03_visual_question_answering/framework_lab.py
{
  "device": "cpu",
  "epochs": 180,
  "question_accuracy": 1.0,
  "overall_accuracy": 1.0,
  "answer_type_accuracy": {
    "yes/no": 1.0,
    "color": 1.0,
    "count": 1.0
  }
}
```
실행 후에는 answer-type accuracy SVG, scratch/framework metrics JSON, 그리고 실행별 해석 리포트가 모두 `artifacts/` 아래에 남는다. 즉 VQA를 단순한 “정답률 하나”가 아니라 **질문 유형별 성능과 오류 원인**으로 읽게 만든다.

## 이 단위에서 특히 볼 것
- overall accuracy가 높아 보여도 `count` 질문이 흔들리면 grounded reasoning이 아직 약하다는 신호다.
- yes/no 질문은 language prior만으로도 잘 맞는 것처럼 보일 수 있어, answer type별 분해가 특히 중요하다.
- `vqa_answer_type_accuracy.svg`를 보면 어떤 answer type이 병목인지 바로 드러난다.
- 행 단위 예측 기록을 보면 “이미지를 봐서 틀렸는지 / 질문을 잘못 읽었는지 / prior에 끌렸는지”를 빠르게 분류할 수 있다.

## 다음 단위와의 연결
이 감각이 있어야 이후 더 큰 multimodal benchmark에서 **overall accuracy + answer type breakdown + qualitative failure case**를 함께 읽을 수 있다. retrieval와 captioning에서 쌓은 grounding 감각이 VQA 해석으로 어떻게 이어지는지 확인하는 마지막 연결 단위다.
