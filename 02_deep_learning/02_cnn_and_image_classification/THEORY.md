# 02 CNN and Image Classification 이론 노트

## 핵심 개념

### 1. local receptive field: 출력 하나가 입력 전체를 보지 않는다
- CNN의 출력 위치 하나는 보통 입력의 작은 patch만 본다.
- 이것이 **local receptive field** 다.
- fully connected layer는 출력 하나가 입력 전체 feature와 곧바로 연결될 수 있지만, convolution은 먼저 작은 주변 문맥만 본다.
- 이미지에서는 이 제한이 약점이 아니라 장점인 경우가 많다. 에지, 코너, 줄무늬 같은 시각 패턴은 대개 **작은 이웃 영역** 에서 먼저 드러나기 때문이다.

### 2. convolution: 작은 pattern detector가 슬라이딩한다
- convolution은 작은 kernel을 이미지 위로 움직이며 weighted sum을 계산한다.
- 그래서 convolution을 “이미지 전체를 한 번에 이해하는 연산”보다 **어떤 작은 패턴이 이 patch 안에 있는가를 점수화하는 규칙**으로 읽는 편이 학습자에게 더 자연스럽다.
- vertical detector는 세로 줄무늬에, horizontal detector는 가로 줄무늬에 더 크게 반응할 수 있다.
- 같은 kernel이 여러 위치에서 반복 사용되므로, 특정 패턴이 왼쪽에 있든 오른쪽에 있든 비슷한 규칙으로 볼 수 있다.

### 3. parameter sharing: 위치가 달라도 같은 규칙을 쓴다
- convolution kernel은 한 위치에만 쓰고 버리는 가중치가 아니라, 이미지 전체 위치에서 반복 재사용된다.
- 이 성질이 **parameter sharing** 이다.
- 결과적으로 “세로 줄무늬가 어디에 있든 같은 detector가 반응한다”는 직관이 생긴다.
- 파라미터 수도 fully connected baseline보다 훨씬 적게 유지할 수 있다.

### 4. pooling: 중요한 반응은 남기고 해상도는 줄인다
- pooling은 feature map의 작은 영역을 하나의 값으로 요약한다.
- max pooling은 가장 강한 반응을 남겨 “그 패턴이 이 근처에 있었는가”를 더 압축해서 전달한다.
- 이 과정에서 정확한 좌표 일부는 버려지므로, pooling은 **정보 보존이 아니라 선택적 요약** 으로 이해해야 한다.
- 대신 계산량이 줄고, 작은 위치 이동에 조금 더 둔감한 표현을 만들 수 있다.

### 5. channel과 feature map은 서로 다른 개념이다
- 입력 channel은 RGB처럼 **원래 데이터가 가진 관측 축** 이다.
- 출력 feature map 수는 모델이 준비한 **서로 다른 detector 개수** 에 가깝다.
- 예를 들어 입력이 3채널이라도 출력 feature map은 2개, 8개, 32개 등 원하는 수로 둘 수 있다.
- 즉 “입력 channel 수”와 “출력 feature map 수”는 같은 숫자일 필요도, 같은 의미일 필요도 없다.

### 6. simple image classification baseline: detector 평균이 곧 class score가 될 수 있다
- toy CNN에서는 convolution → ReLU → pooling 뒤에 남은 feature map 평균값을 class score처럼 읽을 수 있다.
- vertical detector 평균이 horizontal detector 평균보다 크면 vertical class 쪽으로 기울었다고 해석할 수 있다.
- 실제 큰 모델은 보통 여러 conv block 뒤에 더 복잡한 head를 붙이지만, 가장 작은 baseline에서는 이런 방식만으로도 **중간 표현 → class score** 흐름을 설명할 수 있다.

## Common Confusion
- convolution을 “이미지 전체를 한 번에 보는 연산”으로 오해하는 실수
- receptive field가 작으면 무조건 표현력이 부족하다고 단정하는 실수
- 입력 channel과 출력 feature map 수를 같은 의미로 보는 실수
- pooling이 중요한 정보를 전혀 잃지 않는다고 과신하는 실수
- feature map 하나가 곧 class 하나라고 섣불리 연결하는 실수

## 실행에서 꼭 확인할 것
- feature map shape가 입력보다 왜 작아졌는가?
- 같은 detector가 서로 다른 위치의 막대 패턴에 반복 반응하는가?
- pooling 뒤에는 4×4가 2×2로 줄어들며 어떤 정보가 더 요약되는가?
- 입력 channel 수는 3인데 출력 feature map 수는 2라는 사실을 어떻게 설명할 수 있는가?
- pooled score 평균만으로도 toy image classification baseline을 만들 수 있는가?

## 실행 결과 예시
```text
scratch metrics
- dataset_shape: [4, 3, 6, 6]
- local_receptive_field: [3, 3]
- conv_kernel_shape: [2, 3, 3, 3]
- feature_map_shape: [4, 2, 4, 4]
- pooled_shape: [4, 2, 2, 2]
- parameter_sharing_reuse_count: 16
- classification_accuracy: 1.0

framework metrics
- backend: pytorch
- device: cpu
- conv_weight_shape: [2, 3, 3, 3]
- feature_map_shape: [4, 2, 4, 4]
- pooled_shape: [4, 2, 2, 2]
- logits_shape: [4, 2]
- accuracy: 1.0
```

이 숫자는 “작은 patch를 읽는 detector → pooling으로 요약된 feature map → 간단한 class score baseline”이라는 CNN 기본 흐름이 실제 관측으로도 이어진다는 점을 보여 준다.
