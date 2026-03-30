# 01 Tensor Shapes 이론 노트

## 핵심 개념
- **shape**는 텐서가 각 축(axis)에서 몇 개의 원소를 가지는지 나타낸다.
- **batch 차원**은 여러 샘플을 한 번에 처리하기 위해 앞쪽에 두는 축이다.
- **broadcasting**은 길이가 1인 축을 자동 확장해 연산하는 규칙이다.
- **matrix multiplication (matmul)** 은 안쪽 차원이 맞을 때만 가능하다.

## 수식 / 직관
- `(m, n) @ (n, p) -> (m, p)`
- `(2, 3) @ (3, 4) -> (2, 4)`처럼 가운데 차원 `3`이 맞아야 한다.
- batch가 붙으면 `(batch, hidden)` 형태로 읽고, 마지막 축이 feature/channel 역할을 하는지 먼저 본다.

## Common Confusion
- shape의 각 숫자를 순서 없이 읽는 실수
- batch 차원과 feature 차원을 뒤바꾸는 실수
- broadcasting이 자동으로 되므로 “맞는 연산”이라고 착각하는 실수
- matmul과 element-wise 연산의 shape 규칙을 섞어 기억하는 실수
