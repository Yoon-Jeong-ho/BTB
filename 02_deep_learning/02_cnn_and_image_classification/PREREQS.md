# 02 CNN and Image Classification 선행 개념

## 꼭 알고 오면 좋은 것
- `(batch, channel, height, width)` shape를 자연스럽게 읽는 습관
- weighted sum / dot product가 작은 patch 점수 계산으로 이어진다는 감각
- activation, logits, class prediction이 각각 다른 역할이라는 점
- `02_deep_learning/01_perceptron_and_mlp`에서 본 작은 neural baseline 읽기
- max / average 같은 요약 연산이 정보를 압축한다는 기본 이해

## 먼저 다시 보면 좋은 단위
- [00_foundations/01_tensor_shapes](../../00_foundations/01_tensor_shapes/README.md) — 이미지/배치 tensor shape 읽기 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — activation과 class score 구분
- [02_deep_learning/01_perceptron_and_mlp](../01_perceptron_and_mlp/README.md) — 작은 neural baseline과 classification 읽기

## 빠른 자기 점검
- local receptive field를 “출력 하나가 입력의 작은 patch만 본다”는 말로 설명할 수 있는가?
- 같은 kernel이 여러 위치에 재사용된다는 뜻을 parameter sharing 관점에서 말할 수 있는가?
- pooling이 왜 해상도 감소와 정보 요약을 동시에 의미하는지 설명할 수 있는가?
- 입력 channel과 출력 feature map 수를 서로 다른 개념이라고 말할 수 있는가?
- feature map 평균값을 class score baseline처럼 읽을 수 있다는 아이디어를 받아들일 준비가 되었는가?

## 이번 실습에 들어가기 전 팁
- scratch에서는 “직접 계산해 보는 detector”에 집중하고, framework에서는 shape와 tensor 흐름이 같은지 확인한다.
- 숫자를 모두 외우기보다, **작은 patch → feature map → pooling → class score** 흐름을 먼저 잡아도 충분하다.
