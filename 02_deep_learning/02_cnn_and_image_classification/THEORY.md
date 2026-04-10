# 02 CNN and Image Classification 이론 노트

## 핵심 개념

### 1. 왜 이미지에 CNN이 잘 맞는가
- 이미지는 "픽셀 값 목록"이기도 하지만, 더 중요하게는 **가까운 픽셀끼리 함께 의미를 만드는 2차원 구조**다.
- flatten 뒤 MLP로 바로 읽으면 인접한 픽셀 관계와 위치 주변 문맥이 쉽게 사라진다.
- CNN은 이 문제를 **local receptive field + parameter sharing** 으로 다룬다.
- 직관적으로는 "전체 이미지를 한 번에 읽는 큰 규칙 하나"보다, "작은 패턴을 여러 위치에서 반복해서 찾는 규칙"에 가깝다.

### 2. convolution을 어떻게 이해할까
- convolution layer는 작은 kernel(filter)을 이미지 위로 슬라이딩하며 각 위치마다 weighted sum을 계산한다.
- 그래서 convolution은 "이 주변 patch가 내가 찾는 패턴과 얼마나 닮았는가"를 점수화하는 연산처럼 읽을 수 있다.
- 예를 들어 어떤 kernel은 수평선, 어떤 kernel은 세로 경계, 어떤 kernel은 특정 질감 같은 패턴에 더 크게 반응할 수 있다.
- 실제 프레임워크 구현은 엄밀한 수학적 convolution보다 cross-correlation 형태로 설명되는 경우가 많지만, 학습자 입장에서는 "작은 패턴 탐지기"라는 직관이 더 중요하다.

### 3. local receptive field 직관
- local receptive field는 한 출력 위치가 입력의 **작은 주변 영역**만 본다는 뜻이다.
- fully connected layer는 출력 하나가 입력 전체와 곧바로 연결될 수 있지만, convolution은 먼저 작은 이웃만 본다.
- 이 제한은 표현력을 줄이는 것처럼 보일 수 있지만, 이미지에서는 오히려 강한 inductive bias가 된다.
- 이유는 가장자리, 점, 곡선, 질감 같은 시각 패턴이 보통 **국소 영역(local region)** 에서 먼저 나타나기 때문이다.
- 층을 여러 번 쌓으면 초반에는 작은 local pattern을, 뒤로 갈수록 더 큰 receptive field를 통해 더 큰 형태를 본다고 이해할 수 있다.

### 4. parameter sharing이 주는 효과
- 같은 kernel 가중치를 이미지 모든 위치에 반복 사용한다는 것이 parameter sharing이다.
- 덕분에 "왼쪽 위에서 본 선 패턴"과 "오른쪽 아래에서 본 선 패턴"을 같은 규칙으로 감지할 수 있다.
- 이는 파라미터 수를 줄이고, 위치가 조금 이동해도 비슷한 패턴을 재사용할 수 있게 만든다.
- 그래서 CNN은 MLP보다 공간 구조에 더 맞는 기본 가정을 가진다.

### 5. pooling은 무엇을 하나
- pooling은 보통 작은 영역의 값을 요약해 feature map의 해상도를 줄인다.
- max pooling은 "가장 강하게 켜진 반응"을 남기고, average pooling은 "전체 평균 반응"을 남긴다.
- 직관적으로 pooling은 세부 좌표를 일부 버리는 대신, 중요한 반응이 있었는지를 더 압축해서 다음 층으로 넘긴다.
- stride를 크게 둔 convolution도 해상도를 줄인다는 점에서는 비슷해 보이지만, pooling은 보통 "요약"에 더 가깝고 stride convolution은 "학습된 downsampling"에 더 가깝다.
- 따라서 pooling은 계산량을 줄이고 위치 변화에 조금 더 둔감한 표현을 만들지만, 지나치면 작은 위치 정보까지 잃을 수 있다.

### 6. channel과 feature map을 어떻게 읽을까
- 입력 image의 channel은 보통 RGB처럼 **원래 데이터가 가진 관측 축** 이다.
- convolution의 출력 channel 수는 모델이 학습한 **서로 다른 패턴 탐지기 개수**처럼 읽을 수 있다.
- 각 출력 channel은 하나의 feature map을 만든다.
- feature map은 "이 class다"를 직접 말하는 표가 아니라, 특정 패턴이 어느 위치에서 얼마나 강하게 나타났는지 보여 주는 중간 표현이다.
- 초반 layer의 feature map은 에지/코너/질감 같은 저수준 패턴에 민감하고, 뒤쪽으로 갈수록 더 큰 조합 패턴에 민감해질 수 있다.

### 7. 이미지 분류 출력은 어떻게 읽을까
- 여러 convolution / pooling 층을 지난 뒤에는 feature map들을 더 압축하거나 모아 classification head로 보낸다.
- 마지막 선형층의 출력은 보통 **class logits** 이다.
- logits는 아직 확률이 아니라 class별 점수이며, 가장 큰 logit이 예측 class 후보가 된다.
- 중요한 점은 중간 feature map activation과 최종 logits를 섞어 읽지 않는 것이다.
  - feature map: 어디에서 어떤 패턴이 보였는가
  - logits: 최종적으로 어떤 class를 더 지지하는가
- 따라서 분류를 읽을 때는 "어떤 중간 패턴이 누적되어 이 class 점수로 이어졌는가"를 생각해야 한다.

## Common Confusion
- convolution을 "이미지 전체를 한 번에 보는 연산"으로 오해하는 실수
- receptive field가 작으면 모델이 무조건 약하다고 생각하는 실수
- input channel 수와 output feature map 수를 같은 의미로 보는 실수
- pooling이 중요한 정보를 모두 보존한다고 과신하는 실수
- feature map 하나가 곧 class 하나라고 착각하는 실수
- softmax 전 logits와 softmax 후 probability를 구분하지 않는 실수
- CNN이면 위치 이동에 완전히 무감각하다고 과장하는 실수

## 이 단위에서 무엇을 관찰할 것인가
- kernel 하나가 서로 다른 위치를 훑을 때 비슷한 패턴에 반복 반응하는가?
- convolution 뒤 feature map shape가 입력보다 어떻게 달라지는가?
- pooling이나 stride 이후 해상도가 줄어들 때 어떤 정보가 덜 정밀해지는가?
- 출력 channel 수를 늘리면 "서로 다른 패턴 보기"가 어떤 의미인지 감이 오는가?
- feature map이 강하게 반응한 위치와 최종 logits 사이를 어떻게 연결해 설명할 수 있는가?
- flatten-only baseline과 비교했을 때 CNN이 공간 구조를 더 자연스럽게 다룬다는 말을 어떤 관찰로 뒷받침할 수 있는가?
