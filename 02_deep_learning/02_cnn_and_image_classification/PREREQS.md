# 02 CNN and Image Classification 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, channel, height, width)` 또는 `(height, width, channel)` 같은 이미지 텐서 shape 읽기
- dot product / weighted sum이 작은 patch 점수 계산으로도 이어질 수 있다는 감각
- activation, logits, cross entropy가 분류 head에서 각각 무엇을 의미하는지
- `02_deep_learning/01_perceptron_and_mlp`에서 본 MLP 기본 구조와 한계
- class prediction과 intermediate representation을 같은 것으로 보면 안 된다는 점

## 먼저 다시 보면 좋은 단위
- [00_foundations/01_tensor_shapes](../../00_foundations/01_tensor_shapes/README.md) — 이미지/배치 텐서 shape 읽기 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — logits, softmax, cross entropy 역할 복습
- [02_deep_learning/01_perceptron_and_mlp](../01_perceptron_and_mlp/README.md) — fully connected baseline과 nonlinearity 직관 연결

## 빠른 자기 점검
- 이미지를 flatten해서 MLP에 넣는 방식이 가까운 픽셀 관계를 왜 잘 보존하지 못하는지 설명할 수 있는가?
- "작은 kernel이 이미지 위를 슬라이딩한다"는 말을 weighted sum 관점에서 말로 풀 수 있는가?
- 입력 channel 수(RGB)와 출력 feature map 수(learned filters)를 서로 다른 개념으로 설명할 수 있는가?
- pooling이 계산량을 줄이면서도 어떤 정보를 일부 버린다는 점을 이해하는가?
- 최종 logits와 중간 feature map이 각각 무엇을 의미하는지 구분해서 말할 수 있는가?
