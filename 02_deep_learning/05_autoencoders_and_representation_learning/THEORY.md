# 05 Autoencoders and Representation Learning 이론 노트

## 핵심 개념

### 1. reconstruction objective는 무엇을 시키는가
- autoencoder의 가장 기본 목표는 입력 `x`를 받아 다시 `x_hat`로 복원하게 만드는 것이다.
- 보통은 `L(x, x_hat)` 형태의 reconstruction loss를 최소화한다.
- 중요한 점은 이 목표가 단순히 "정답 label 맞히기"가 아니라, **입력 구조를 유지하는 표현을 만들기** 를 요구한다는 것이다.
- 그래서 supervised target이 없더라도 학습 신호를 만들 수 있다.

### 2. encoder / latent / decoder는 각각 무엇을 하는가
- **encoder** 는 입력을 더 작거나 더 구조화된 표현으로 바꾸는 쪽이다.
- **latent representation `z`** 는 입력 전체를 그대로 들고 가지 못하는 요약 공간이다.
- **decoder** 는 latent를 바탕으로 원래 입력을 다시 복원하려 한다.
- 직관적으로는 encoder가 "요약 규칙"을 만들고, decoder는 "이 요약만 보고 어디까지 복원 가능한가"를 시험하는 역할이다.

### 3. bottleneck intuition: 왜 일부러 좁혀야 하는가
- latent dimension이 입력과 거의 같고 제약도 약하면, 모델은 입력을 거의 복사하는 방향으로 쉽게 흐를 수 있다.
- 반대로 **bottleneck** 을 두면 모델은 모든 세부값을 그대로 넘길 수 없으므로, 상대적으로 중요한 패턴을 압축해서 담아야 한다.
- 하지만 bottleneck이 너무 좁으면 필요한 정보까지 사라져 reconstruction이 지나치게 나빠질 수 있다.
- 즉 좋은 bottleneck은 "복원은 가능하되, 쓸모없는 세부 복사는 줄이는 정도"를 찾는 문제와 연결된다.

### 4. denoising / compression variants는 무엇이 다른가
- **denoising autoencoder** 는 noisy input `x_noisy`를 넣고 원래 clean input `x`를 복원하게 한다.
- 이 설정은 단순 픽셀 복사보다, 입력의 더 안정적인 구조를 잡도록 압박한다.
- **compression-oriented autoencoder** 는 bottleneck 크기, sparsity, quantization 같은 제약을 통해 저장 효율이나 압축률에 더 초점을 둔다.
- 둘 다 reconstruction objective를 쓰지만, denoising은 "노이즈를 넘어서 구조를 회복하는 능력"을, compression은 "적은 code 길이로 핵심 정보를 얼마나 보존하는가"를 더 강하게 본다.

### 5. 왜 latent representation이 중요한가
- 좋은 latent는 단지 복원용 중간값이 아니라, 입력의 중요한 구조를 더 다루기 쉬운 좌표계로 옮긴 결과일 수 있다.
- 그래서 latent space는 retrieval / nearest neighbor 검색, clustering, anomaly detection, compact feature, generative modeling의 기반으로 이어진다.
- 즉 autoencoder의 핵심 질문은 "복원이 되느냐"에서 끝나지 않고, **그 과정에서 얻은 latent가 다른 작업에도 의미가 있느냐** 로 확장된다.

## Common Confusion
- reconstruction loss가 낮으면 항상 좋은 representation이라고 단정하는 실수
- bottleneck이 작을수록 무조건 더 "의미 있는" latent가 된다고 생각하는 실수
- autoencoder가 입력을 복원하므로 그냥 identity mapping과 다를 바 없다고 넘기는 실수
- denoising autoencoder를 단순 데이터 증강 정도로만 이해하는 실수
- latent dimension을 줄였다는 사실만으로 압축/일반화가 자동 보장된다고 보는 실수

## 실행에서 꼭 확인할 것
- latent dimension을 1, 2, 3으로 바꿨을 때 reconstruction mse가 어떻게 줄어드는가?
- encoder / latent / decoder 역할을 metrics의 어떤 필드로 다시 읽을 수 있는가?
- noisy input 대비 denoised mse가 얼마나 줄어드는가?
- compression ratio가 1보다 작을 때, 실제 reconstruction 손실과 어떤 trade-off를 보이는가?

## 실행 결과 예시
```text
scratch metrics
- input_dim: 8
- bottleneck_dims_compared: [1, 2, 3]
- bottleneck_results.1.reconstruction_mse: 0.0925
- bottleneck_results.3.reconstruction_mse: 0.0
- denoising_variant.raw_noisy_mse: 0.00434219
- denoising_variant.denoised_mse: 0.0
- figure_path: artifacts/scratch-manual/autoencoder_bottleneck.svg

framework metrics
- device: cpu
- compression_autoencoder.latent_dim: 3
- compression_autoencoder.final_loss: 0.0
- narrow_bottleneck_autoencoder.latent_dim: 1
- narrow_bottleneck_autoencoder.final_loss: 0.04680542
- denoising_autoencoder.final_loss: 0.0
- denoising_autoencoder.raw_noisy_baseline_loss: 0.00434219
```
이 숫자는 reconstruction objective가 단순 복사가 아니라 **어떤 latent bottleneck을 두고도 구조를 되살릴 수 있는가** 를 묻는다는 점, 그리고 denoising / compression variant가 같은 autoencoder 구조를 서로 다른 압력으로 사용한다는 점을 다시 보여 준다.
