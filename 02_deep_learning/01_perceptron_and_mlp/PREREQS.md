# 01 Perceptron and MLP 선행 개념

## 꼭 알고 오면 좋은 것
- dot product와 weighted sum이 feature 여러 개를 하나의 점수로 압축한다는 감각
- bias가 decision boundary를 평행 이동시킨다는 기본 이해
- activation과 loss가 서로 다른 역할이라는 점
- gradient / backpropagation이 파라미터를 업데이트하는 방향 신호라는 점
- 작은 baseline을 먼저 세우고 비교하는 습관

## 먼저 다시 보면 좋은 단위
- [00_foundations/01_tensor_shapes](../../00_foundations/01_tensor_shapes/README.md) — 벡터, 행렬곱, shape 읽기 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — sigmoid/ReLU와 loss의 역할 구분
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — gradient 신호가 어떻게 업데이트로 이어지는지 복습
- [01_ml/01_tabular_classification](../../01_ml/01_tabular_classification/README.md) — 가장 작은 분류 baseline을 먼저 읽는 습관 연결

## 빠른 자기 점검
- `w·x + b`가 0보다 큰지 작은지로 분류한다는 말을 2차원 점 그림으로 설명할 수 있는가?
- "직선 하나로는 XOR를 못 푼다"는 말을 말로 풀 수 있는가?
- linear layer를 두 번 쌓아도 activation이 없으면 결국 선형 변환 하나와 같다는 이유를 설명할 수 있는가?
- 데이터가 잘 안 풀릴 때 optimizer 문제와 표현력 부족 문제를 구분해서 의심해야 한다는 말을 이해하는가?

## 이번 실습에 들어가기 전 팁
- scratch에서는 규칙을 먼저 읽고, framework에서는 그 규칙이 실제 모델 성능 차이로 어떻게 드러나는지 본다.
- 정확한 공식을 모두 외우기보다, "직선 하나냐 / hidden layer가 있느냐"를 먼저 구분해도 충분하다.
