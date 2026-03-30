# 02 Activation and Loss 선행 개념

## 꼭 알고 오면 좋은 것
- tensor shape를 앞에서부터 읽는 습관
- logits와 probability가 같은 말이 아니라는 점
- 분류(classification)와 회귀(regression)에서 비교 방식이 다르다는 점

## 빠른 자기 점검
- softmax를 적용한 뒤 각 행의 합이 왜 1이 되는지 한 문장으로 설명할 수 있는가?
- binary target 하나와 class index target 하나가 서로 다른 loss를 요구하는 이유를 말할 수 있는가?
- ReLU가 음수를 0으로 만드는 것이 왜 “오류”가 아닌지 설명할 수 있는가?
