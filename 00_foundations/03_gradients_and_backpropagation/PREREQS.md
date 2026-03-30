# 03 Gradients and Backpropagation 선행 개념

## 꼭 알고 오면 좋은 것
- tensor shape와 batch 차원을 읽는 기본 감각
- activation / logits / loss를 구분하는 습관
- 1차 함수와 제곱 오차를 식으로 읽는 최소한의 미분 감각

## 빠른 자기 점검
- `prediction = w * x + b`에서 `w`가 커지면 prediction이 어떻게 변하는지 설명할 수 있는가?
- loss가 scalar 하나로 줄어드는 이유를 한 문장으로 말할 수 있는가?
- “analytic gradient”와 “finite-difference gradient”의 차이를 감으로라도 구분할 수 있는가?
