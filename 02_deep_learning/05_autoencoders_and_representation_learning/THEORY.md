# 05 Autoencoders and Representation Learning 이론 노트

## 핵심 개념

### 1. reconstruction objective는 무엇을 시키는가
- autoencoder의 가장 기본 목표는 입력 `x`를 받아 다시 `x_hat`로 복원하게 만드는 것이다.
- 보통은 `L(x, x_hat)` 형태의 reconstruction loss를 최소화한다.
  - 연속값 입력이면 MSE 같은 손실
  - 0~1 범위 픽셀이면 BCE 계열 손실
- 중요한 점은 이 목표가 단순히 "정답 label 맞히기"가 아니라, **입력 구조를 유지하는 표현을 만들기** 를 요구한다는 것이다.
- 그래서 supervised target이 없더라도 학습 신호를 만들 수 있다.

### 2. encoder / latent / decoder는 각각 무엇을 하는가
- **encoder** 는 입력을 더 작거나 더 구조화된 표현으로 바꾸는 쪽이다.
  - `z = f_encoder(x)`
- **latent representation `z`** 는 입력 전체를 그대로 들고 가지 못하는 요약 공간이다.
  - 여기서 어떤 축이 생기고 어떤 정보가 섞이는지가 representation learning의 핵심이다.
- **decoder** 는 latent를 바탕으로 원래 입력을 다시 복원하려 한다.
  - `x_hat = f_decoder(z)`
- 직관적으로는 encoder가 "요약 규칙"을 만들고, decoder는 "이 요약만 보고 어디까지 복원 가능한가"를 시험하는 역할이다.

### 3. bottleneck intuition: 왜 일부러 좁혀야 하는가
- latent dimension이 입력과 거의 같고 제약도 약하면, 모델은 입력을 거의 복사하는 방향으로 쉽게 흐를 수 있다.
- 반대로 **bottleneck** 을 두면 모델은 모든 세부값을 그대로 넘길 수 없으므로, 상대적으로 중요한 패턴을 압축해서 담아야 한다.
- 이때 bottleneck은 "모델 괴롭히기"가 아니라 **무엇이 중요한 구조인가를 강제로 드러내는 장치** 로 볼 수 있다.
- 하지만 bottleneck이 너무 좁으면 필요한 정보까지 사라져 reconstruction이 지나치게 나빠질 수 있다.
- 즉 좋은 bottleneck은 "복원은 가능하되, 쓸모없는 세부 복사는 줄이는 정도"를 찾는 문제와 연결된다.

### 4. denoising / compression variants는 무엇이 다른가
- **vanilla autoencoder** 는 입력을 그대로 넣고 그대로 복원한다.
- **denoising autoencoder** 는 노이즈가 섞인 입력 `x_noisy`를 넣고 원래 깨끗한 입력 `x`를 복원하게 한다.
  - 이 설정은 단순 픽셀 복사보다, 입력의 더 안정적인 구조를 잡도록 압박한다.
- **compression-oriented autoencoder** 는 bottleneck 크기, sparsity, quantization 같은 제약을 통해 저장 효율이나 압축률에 더 초점을 둔다.
- 둘 다 reconstruction objective를 쓰지만, denoising은 "노이즈를 넘어서 구조를 회복하는 능력"을, compression 쪽은 "적은 코드 길이로 얼마나 핵심 정보를 보존하는가"를 더 강하게 본다.

### 5. 왜 latent representation이 중요한가
- 좋은 latent는 단지 복원용 중간값이 아니라, 입력의 중요한 구조를 더 다루기 쉬운 좌표계로 옮긴 결과일 수 있다.
- 그래서 latent space는 다음 같은 downstream 활용으로 이어진다.
  - 비슷한 샘플끼리 가까이 두는 retrieval / nearest neighbor 검색
  - clustering, anomaly detection 같은 비지도 탐색
  - classifier나 regressor에 넣는 compact feature
  - VAE, diffusion, multimodal alignment처럼 더 큰 representation learning / generative modeling의 기반
- 즉 autoencoder의 핵심 질문은 "복원이 되느냐"에서 끝나지 않고, **그 과정에서 얻은 latent가 다른 작업에도 의미가 있느냐** 로 확장된다.

## 수식 / 직관
- 기본 구조는 아래처럼 쓸 수 있다.
  - `z = f_theta(x)`
  - `x_hat = g_phi(z)`
  - `L_recon = d(x, x_hat)`
- 여기서 `d`는 입력 형태에 따라 달라진다.
- representation learning 관점에서는 `z`가 다음 조건을 어느 정도 만족하는지 함께 본다.
  - 비슷한 입력은 latent에서도 가깝게 모이는가?
  - 노이즈나 불필요한 세부 변화에 너무 민감하지 않은가?
  - downstream task가 더 쉬워지는가?

## Common Confusion
- reconstruction loss가 낮으면 항상 좋은 representation이라고 단정하는 실수
- bottleneck이 작을수록 무조건 더 "의미 있는" latent가 된다고 생각하는 실수
- autoencoder가 입력을 복원하므로 그냥 identity mapping과 다를 바 없다고 넘기는 실수
- denoising autoencoder를 단순 데이터 증강 정도로만 이해하는 실수
- latent dimension을 줄였다는 사실만으로 압축/일반화가 자동 보장된다고 보는 실수

## 이 단위에서 무엇을 관찰할 것인가
- latent dimension을 바꿨을 때 reconstruction error와 표현력 사이의 균형이 어떻게 변하는가?
- noisy input과 clean target을 분리했을 때 decoder가 무엇을 복원하고 무엇을 버리는가?
- latent에서 가까운 샘플들이 입력 공간에서도 실제로 비슷한가?
- reconstruction이 잘 되어도 downstream feature로는 약할 수 있는 경우가 있는가?
- representation learning 관점에서 "복원이 잘 됨"과 "유용한 요약을 얻음" 사이를 어떻게 구분할 수 있는가?
