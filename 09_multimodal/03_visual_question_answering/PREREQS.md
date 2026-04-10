# 03 Visual Question Answering 선행 개념

## 꼭 알고 오면 좋은 것
- retrieval와 captioning에서 봤던 image-text grounding 감각
- classification에서 softmax logits를 answer vocabulary로 읽는 기본 습관
- overall accuracy만으로는 failure mode를 설명하기 어렵다는 점

## 빠른 자기 점검
- yes/no, color, count 같은 answer type이 왜 서로 다른 난도를 가질 수 있는지 설명할 수 있는가?
- 질문 텍스트만 보고 답을 찍는 shortcut bias가 왜 multimodal 모델에서 위험한지 말할 수 있는가?
- count 질문이 흔들릴 때 representation 문제와 reasoning 문제를 어떻게 구분해 볼지 한 문장으로 말할 수 있는가?
