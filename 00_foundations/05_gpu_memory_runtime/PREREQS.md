# 05 GPU Memory Runtime 선행 개념

## 꼭 알고 오면 좋은 것
- tensor shape를 앞에서부터 읽는 습관
- dtype이 원소당 바이트 수에 영향을 준다는 점
- forward와 backward가 서로 다른 단계라는 점

## 빠른 자기 점검
- `(32, 512, 768)` fp32 텐서가 대략 얼마나 큰지 계산해볼 수 있는가?
- 같은 모델에서도 inference보다 training이 더 무겁다고 말할 수 있는 이유를 한 문장으로 설명할 수 있는가?
