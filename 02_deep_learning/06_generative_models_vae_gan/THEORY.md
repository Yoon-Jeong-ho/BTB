# 06 Generative Models: VAE, GAN 이론 노트

## 핵심 개념

### 1. 왜 autoencoder 다음에 generative model을 보는가
- autoencoder는 입력을 압축하고 복원하면서 latent representation을 만든다.
- 하지만 deterministic autoencoder만으로는 latent space의 아무 점이나 골라 새 샘플을 안정적으로 만들기 어렵다.
- 그래서 generative model은 "좋은 요약을 만드는가"에서 더 나아가, **샘플링 가능한 latent 구조를 만들거나**, 혹은 **진짜 같은 샘플을 직접 만들어 내는가**를 본다.
- VAE와 GAN은 이 질문에 대한 대표적인 두 답변이다.

### 2. VAE vs GAN: 큰 그림에서 무엇이 다른가
- **VAE (Variational Autoencoder)**
  - encoder가 latent 분포의 파라미터를 내고, 그 분포에서 샘플링한 `z`로 decoder가 입력을 복원한다.
  - 핵심 질문은 "latent space를 얼마나 샘플링 가능하게 정리할 것인가"에 가깝다.
  - 보통 reconstruction term과 KL divergence term을 함께 본다.
- **GAN (Generative Adversarial Network)**
  - generator는 noise에서 가짜 샘플을 만들고, discriminator는 진짜/가짜를 구분하려 한다.
  - 핵심 질문은 "실제 데이터처럼 보이는 샘플을 얼마나 설득력 있게 만들 것인가"에 가깝다.
  - 보통 min-max 게임, adversarial loss, generator/discriminator 균형이 중요하다.
- 아주 거칠게 말하면 VAE는 **확률적 latent structure**, GAN은 **adversarial sample realism** 쪽에 무게를 둔다.

### 3. latent sampling intuition: VAE는 무엇을 배우는가
- VAE encoder는 입력 하나를 latent 점 하나로 보내는 대신, 보통 `mu`와 `logvar` 같은 **분포 파라미터**를 만든다.
- 여기서 latent sample은 대략 `z = mu + sigma * epsilon` 형태로 얻는다.
  - `epsilon`은 표준 정규분포 노이즈
  - `sigma`는 분산 또는 표준편차에 해당하는 값
- 이 구조 덕분에 VAE는 "입력마다 딱 한 점"이 아니라 **주변 latent 영역 전체**를 다루도록 압박받는다.
- KL term은 각 샘플의 posterior가 prior(보통 표준 정규분포)와 너무 멀어지지 않게 잡아 주며, 그 결과 latent space가 샘플링 가능한 형태로 정리되길 기대한다.
- 그래서 VAE를 볼 때는 reconstruction quality뿐 아니라, **latent traversal이 부드러운가**, **prior에서 샘플링한 점도 그럴듯한가**를 함께 본다.

### 4. adversarial intuition: GAN은 무엇을 배우는가
- GAN의 generator `G(z)`는 noise `z`에서 가짜 샘플을 만든다.
- discriminator `D(x)`는 들어온 샘플이 진짜 데이터인지 generator가 만든 가짜인지 판별하려 한다.
- discriminator가 잘 속지 않도록 generator가 업데이트되고, generator가 좋아지면 discriminator도 더 정교해진다.
- 이 경쟁은 고정된 reconstruction target 없이도 **"더 진짜같이 보이는 방향"**의 신호를 줄 수 있다.
- 그래서 GAN은 VAE보다 샘플이 더 sharp하게 보일 수 있지만, 동시에 학습이 훨씬 불안정해지기 쉽다.
- 관찰의 핵심은 loss 숫자 하나보다, **generator가 실제로 다양한 샘플을 만드는가**, **discriminator가 너무 강하거나 너무 약하지 않은가**다.

### 5. 품질, coverage, stability를 같이 봐야 한다
- VAE는 종종 샘플이 더 smooth하거나 blurry하게 보일 수 있다.
  - 대신 latent interpolation, sample continuity, mode coverage 쪽에서 해석이 쉬운 경우가 있다.
- GAN은 더 sharp한 샘플을 줄 수 있다.
  - 대신 일부 mode만 반복 생성하는 **mode collapse** 위험이 크다.
- 따라서 generative model에서는 다음 세 가지를 함께 봐야 한다.
  1. **quality** — 샘플 하나하나가 얼마나 그럴듯한가?
  2. **coverage/diversity** — 데이터 분포의 여러 mode를 얼마나 고르게 담는가?
  3. **stability** — 학습이 얼마나 일관되게 수렴하고 실패를 읽기 쉬운가?
- 어느 축을 더 우선시하느냐에 따라 VAE와 GAN의 장단점 해석도 달라진다.

### 6. common confusion
- VAE의 KL term을 단순 regularization 추가 정도로만 보고, latent sampling 가능성 확보라는 목적을 놓치는 실수
- VAE sample이 blurry하다는 이유만으로 "항상 GAN보다 나쁘다"고 결론내리는 실수
- GAN loss가 줄어드는지만 보면 학습이 잘된다고 생각하는 실수
- discriminator가 매우 강하면 generator도 자동으로 좋아질 것이라고 믿는 실수
- posterior collapse와 mode collapse를 둘 다 "latent가 망가짐" 정도로만 뭉뚱그리는 실수
- 생성 모델 평가를 accuracy처럼 단일 숫자 하나로 끝내려는 실수

## 자주 헷갈리는 지점

### posterior collapse vs mode collapse
- **posterior collapse (주로 VAE)**
  - decoder가 너무 강하거나 KL pressure가 커서, latent `z`를 거의 안 써도 reconstruction이 되는 경우
  - 결과적으로 `mu`, `logvar`가 prior 근처로 몰리고 latent usage가 약해진다.
- **mode collapse (주로 GAN)**
  - generator가 여러 noise 입력에 대해서도 비슷한 샘플만 반복 생성하는 경우
  - discriminator를 일시적으로 속이는 좁은 전략에 갇혀 diversity가 무너진다.
- 둘 다 "표현 공간이 기대만큼 작동하지 않는다"는 점은 비슷하지만, 원인과 관찰 방법은 다르다.

### reconstruction이 좋아도 generation이 좋은 것은 아니다
- VAE에서 reconstruction loss가 잘 내려가도 prior sampling 결과가 그럴듯하지 않을 수 있다.
- autoencoder 계열에서는 입력 재현과 샘플 생성 사이에 간격이 있다.
- 그래서 input reconstruction만 보고 generative quality를 결론내리면 안 된다.

### GAN loss는 읽기 어렵다
- generator loss와 discriminator loss는 경쟁 구조 안에서 함께 움직인다.
- loss가 예쁘게 내려간다고 항상 샘플이 좋아지는 것도 아니고, loss가 흔들린다고 무조건 실패도 아니다.
- 샘플 inspection, diversity check, update balance를 같이 봐야 한다.

## 이 단위에서 무엇을 관찰할 것인가
- VAE에서 `mu`, `logvar`, sampled `z`, reconstruction의 shape와 역할을 분리해서 읽을 수 있는가?
- KL term의 크기를 바꿨을 때 reconstruction quality와 latent usage 사이의 균형이 어떻게 달라질 것 같은가?
- latent interpolation을 가정했을 때 VAE sample transition이 부드럽게 이어질 것 같은가?
- GAN에서 generator와 discriminator가 어느 쪽으로든 지나치게 우세해질 때 어떤 failure pattern이 생길 것 같은가?
- sharp sample과 diverse sample이 항상 같이 오지 않는다는 점을 샘플 inspection 관점에서 설명할 수 있는가?
- later diffusion model이나 multimodal generation을 볼 때도, latent structure / sample realism / training stability라는 세 축으로 질문을 이어 갈 수 있는가?
