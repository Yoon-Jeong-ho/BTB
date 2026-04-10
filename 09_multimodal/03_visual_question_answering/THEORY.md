# 03 Visual Question Answering 이론 노트

## 핵심 개념
- **visual question answering (VQA)** 은 이미지와 질문을 함께 읽고 정답을 예측하는 multimodal classification 문제로 볼 수 있다.
- **answer type** 은 정답의 형태를 뜻한다. toy unit에서는 `yes/no`, `color`, `count` 세 가지를 다룬다.
- **shortcut bias** 는 모델이 이미지를 충분히 보지 않고, 자주 나오는 답이나 질문 패턴만으로 정답을 추정하는 현상이다.
- **grounded reasoning** 은 질문이 요구하는 시각 단서를 실제 이미지 feature와 연결해 답을 내는 과정이다.
- **multimodal fusion** 은 이미지 표현과 질문 표현을 하나의 예측 공간으로 합치는 단계다.

## 수식 / 직관
- 이미지 특징을 `v`, 질문 표현을 `q` 라고 두면 VQA는 대체로 `P(a | v, q)` 를 분류 문제처럼 학습한다.
- answer vocabulary가 작을 때는 정답 후보 집합 `A` 에 대해 `softmax(W [v; q] + b)` 형태로 볼 수 있다.
- 이때 `q` 만으로도 잘 맞는 질문(예: yes/no)이 많으면, 모델이 `v` 를 덜 보는 shortcut으로 빠질 수 있다.
- 그래서 VQA 해석에서는 overall accuracy 하나보다 **answer type별 accuracy** 와 실패 사례가 더 중요하다.

## 왜 count 질문이 자주 어렵나
- count는 색/존재 여부보다 더 정밀한 시각 구분을 요구한다.
- 이미지 전체를 하나의 전역 feature로 압축하면, “있다/없다”는 남아도 “몇 개인가” 정보는 뭉개지기 쉽다.
- 그래서 tiny 모델에서도 count accuracy가 낮게 나오면, 이는 단순 dataset noise가 아니라 **집계 정보가 representation에서 사라졌을 가능성**을 시사한다.

## 이 단위에서 꼭 볼 것
- scratch 규칙기가 왜 yes/no와 color는 맞히면서 count는 흔들렸는가?
- framework tiny classifier가 count까지 복구했다면, 그것은 이미지 특징과 질문 표현을 어떻게 더 잘 결합했기 때문인가?
- overall accuracy가 같더라도 answer type breakdown이 다르면 어떤 해석 차이가 생기는가?
- qualitative row를 볼 때 `predicted_answer` 와 `error_reason` 을 함께 남기는 이유는 무엇인가?

## Common Confusion
- VQA를 단순한 text classification처럼 보고 이미지를 부차적인 입력으로 취급하는 실수
- yes/no accuracy가 높다고 해서 grounded reasoning도 높다고 오해하는 실수
- count failure를 “데이터가 작아서 그런가 보다”로 넘기고, answer type breakdown을 보지 않는 실수
- captioning과 달리 출력이 짧다는 이유로 qualitative inspection이 덜 중요하다고 생각하는 실수

## PyTorch tiny VQA demo에서 보는 구조
- 이 unit의 `framework_lab.py`는 대형 VLM이 아니라, **작은 이미지 projection + 질문 token embedding 평균 + MLP 분류기** 조합으로 VQA의 핵심 흐름만 재현한다.
- 즉 image feature는 시각 단서를, question embedding은 질의 조건을 나타내고, fusion MLP는 둘을 합쳐 정답 vocabulary를 예측한다.
- CPU-safe toy demo라도 “이미지 + 질문 + answer vocabulary + answer type별 해석”이라는 VQA 핵심은 충분히 볼 수 있다.

## 실행 결과 예시
```text
scratch metrics
- overall_accuracy: 0.833333
- answer_type_accuracy.yes/no: 1.0
- answer_type_accuracy.color: 1.0
- answer_type_accuracy.count: 0.5
- figure_path: artifacts/scratch-manual/vqa_answer_type_accuracy.svg

framework metrics
- device: cpu
- question_accuracy: 1.0
- overall_accuracy: 1.0
- answer_type_accuracy.count: 1.0
- loss_history_tail[-1]: 0.001 이하
```
이 숫자는 “정답률이 올랐다”를 넘어서, **어떤 answer type에서 개선이 일어났는지, count 같은 grounded reasoning 질문이 회복되었는지**를 함께 읽어야 VQA를 제대로 해석할 수 있음을 보여 준다.
