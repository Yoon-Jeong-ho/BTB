# 06 Generative Models: VAE, GAN 이론 노트

## 핵심 개념

### 1. autoencoder 다음에 왜 generative model을 보는가
- deterministic autoencoder는 입력을 잘 압축하고 복원할 수 있어도, latent space의 아무 점이나 찍어서 **새 샘플을 안정적으로 만들기** 는 어렵다.
- 그래서 생성 모델은 “representation을 만들었다”에서 한 걸음 더 나아가, **샘플링 가능한 latent space를 만들 것인가**, 혹은 **진짜처럼 보이는 샘플을 직접 밀어 올릴 것인가** 를 묻는다.
- VAE와 GAN은 이 두 질문에 대한 대표적인 출발점이다.

### 2. VAE: latent를 분포로 다루는 방식
- VAE encoder는 입력 하나를 latent 점 하나로 보내는 대신, 보통 `mu`와 `logvar` 같은 **분포 파라미터**를 만든다.
- 여기서 `z = mu + sigma * epsilon` 으로 샘플링하는데, 이 과정을 가능하게 하는 핵심이 **reparameterization trick** 이다.
- reconstruction term은 입력을 다시 맞히라고 요구하고, KL term은 posterior가 prior에서 너무 멀어지지 않게 잡아 준다.
- 그래서 VAE를 읽을 때는 reconstruction quality뿐 아니라, interpolation이 부드러운가, prior sample도 그럴듯한가, latent usage가 실제로 남아 있는가를 같이 봐야 한다.

### 3. posterior collapse: VAE의 대표 실패
- decoder가 너무 강하거나 KL 압력이 과도하면, decoder가 `z` 없이도 reconstruction을 해버릴 수 있다.
- 이때 `mu`가 0 근처로 몰리고, KL이 너무 작아지며, latent usage가 줄어든다. 이것이 **posterior collapse** 다.
- reconstruction 숫자만 보면 겉으로 멀쩡해 보일 수도 있어서, KL / latent usage / interpolation 품질을 같이 확인해야 한다.

### 4. GAN: adversarial game으로 sample realism을 끌어올리는 방식
- GAN의 generator는 noise에서 가짜 샘플을 만들고, discriminator는 그 샘플이 진짜인지 가짜인지 구분하려 한다.
- 이 구조는 reconstruction target 없이도 **adversarial** 신호로 sample realism을 밀어 올릴 수 있다.
- 그래서 GAN은 종종 VAE보다 sharper sample을 줄 수 있지만, 동시에 학습 안정성과 해석 가능성은 더 나빠질 수 있다.
- GAN을 볼 때는 generator loss / discriminator loss뿐 아니라, batch diversity, mode coverage, sample inspection을 함께 봐야 한다.

### 5. mode collapse: GAN의 대표 실패
- generator가 다양한 noise를 받아도 비슷한 샘플만 반복 생성하는 현상이 **mode collapse** 다.
- 이때 개별 샘플은 꽤 진짜같아 보일 수 있어서, loss만 보면 문제를 놓치기 쉽다.
- 따라서 GAN에서는 “샘플 한 장이 그럴듯한가”와 함께 “분포의 여러 mode를 덮는가”를 반드시 같이 확인해야 한다.

### 6. VAE vs GAN을 한 줄로 대비하면
- VAE는 **샘플링 가능한 latent geometry** 를 만들려 하고,
- GAN은 **adversarial sample realism** 을 밀어 올리려 한다.
- 그래서 VAE는 interpolation / KL / latent usage 해석이 강하고, GAN은 sharp sample / realism 쪽이 강하지만 collapse 위험이 더 크다.

## Common Confusion
- VAE의 KL term을 단순 regularization 추가 정도로만 보고, 샘플링 가능한 latent space를 만들려는 목적을 놓치는 실수
- VAE reconstruction이 좋으면 generative sample도 자동으로 좋다고 생각하는 실수
- GAN loss가 예쁘게 움직이면 diversity도 자동으로 좋아진다고 생각하는 실수
- posterior collapse와 mode collapse를 둘 다 “latent가 망가짐” 정도로만 뭉뚱그리는 실수
- generative model을 accuracy처럼 단일 숫자 하나로 끝내려는 실수

## 실행에서 꼭 확인할 것
- `mu`, `logvar`, sampled `z`, reconstruction이 각각 어떤 역할을 하는가?
- reparameterization trick이 없다면 왜 encoder가 sampling을 학습하기 어려운가?
- posterior collapse probe에서 latent usage와 reconstruction은 어떻게 같이 변하는가?
- GAN balanced generator와 collapsed generator의 mode coverage 차이는 얼마나 나는가?
- adversarial loss가 비슷해 보여도 pairwise diversity / coverage가 왜 더 중요한가?

## 실행 결과 예시
```text
scratch metrics
- vae.reconstruction_mse: 0.176561
- vae.kl_term: 1.637268
- vae.prior_sample_spread: 1.25591
- vae.posterior_collapse_probe.collapsed_latent_usage: 0.0
- gan.balanced_mode_coverage: 4
- gan.collapsed_mode_coverage: 1
- figure_path: artifacts/scratch-manual/vae_gan_diagnostics.svg

framework metrics
- device: cpu
- vae.final_reconstruction_loss: 0.205331
- vae.final_kl_loss: 1.427023
- vae.posterior_usage_mean_abs: 0.644694
- gan.mode_coverage: 4
- gan.loss_only_is_ambiguous: true
- gan.collapsed_probe.mode_coverage: 1
```
이 숫자는 VAE가 latent sampling과 posterior collapse 관찰에 강하고, GAN은 adversarial realism과 mode collapse 관찰에 더 민감하다는 점을 다시 보여 준다.
