# 01 Tensor Shapes 분석

## 관측 결과
- scratch matmul shape: `[2, 4]`
- scratch broadcast result shape: `[2, 3]`
- scratch mismatch error: `operands could not be broadcast together with shapes (2,3) (4,) `
- framework logits shape: `[4, 3]`
- framework probs row sums: `[1.0, 1.0, 1.0, 1.0]`

## 해석
- `2 x 3`과 `3 x 4`의 행렬 곱 결과가 `2 x 4`가 되는 것을 직접 확인했다.
- `(2, 3)` 텐서에 `(2, 1)` 텐서를 더하면 broadcasting으로 `(2, 3)` 결과를 만들 수 있었다.
- 반대로 `(4,)` 벡터를 더하려고 하면 마지막 축 길이가 맞지 않아 shape mismatch가 바로 발생했다.
- PyTorch `Linear(8, 3)`는 `(4, 8)` batch를 `(4, 3)` logits로 바꾸며, softmax 이후 각 행의 확률 합은 1에 가깝다.

## 실패 사례
- mismatch 에러 메시지는 `operands could not be broadcast together with shapes (2,3) (4,) ` 였다.
- 이 에러는 “원소 수가 달라서”가 아니라 broadcasting 규칙상 축 정렬이 맞지 않아서 생긴다.

## 관련 이론
- [THEORY.md](./THEORY.md): shape, broadcasting, batch dimension 핵심 개념을 다시 확인한다.
