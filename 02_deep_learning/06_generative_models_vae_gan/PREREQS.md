# 06 Generative Models: VAE, GAN 선행 개념

## 꼭 알고 오면 좋은 것
- encoder, decoder, latent representation이 각각 어떤 역할을 하는지
- reconstruction loss가 label 없이도 학습 신호가 될 수 있다는 점
- 확률분포의 평균, 분산(또는 표준편차), 샘플링이라는 표현을 아주 기본적으로 읽는 감각
- KL divergence를 "두 분포가 얼마나 다른가" 정도의 직관으로 받아들일 준비
- min-max 최적화나 경쟁적 학습이 왜 일반 supervised training보다 불안정할 수 있는지에 대한 감각
- loss 숫자 하나만 보지 않고 sample quality, diversity, failure pattern을 함께 봐야 한다는 점

## 먼저 다시 보면 좋은 단위
- [02_deep_learning/05_autoencoders_and_representation_learning](../05_autoencoders_and_representation_learning/README.md) — latent representation과 reconstruction objective 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — loss가 학습 신호를 주는 구조 복습
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — 생성 모델도 결국 gradient로 학습된다는 점 복습
- [00_foundations/04_regularization_and_normalization](../../00_foundations/04_regularization_and_normalization/README.md) — regularization/normalization이 학습 안정성과 연결되는 감각 복습
- [02_deep_learning/01_perceptron_and_mlp](../01_perceptron_and_mlp/README.md) — 작은 MLP encoder/generator/discriminator를 상상하기 위한 기본 신경망 감각 복습

## 빠른 자기 점검
- deterministic autoencoder와 "샘플링 가능한 latent model"의 차이를 한두 문장으로 설명할 수 있는가?
- 평균과 분산을 이용해 latent distribution을 표현한다는 말을 들었을 때 겁먹지 않고 읽을 수 있는가?
- KL divergence가 정확한 유도식은 몰라도, 분포를 prior 쪽으로 정리하려는 압력이라는 직관은 있는가?
- generator와 discriminator가 동시에 학습되면 왜 loss interpretation이 더 까다로워지는지 이해하는가?
- 샘플이 선명해 보여도 diversity가 낮을 수 있다는 말을 받아들일 수 있는가?
- posterior collapse와 mode collapse가 서로 다른 실패라는 점을 구분할 준비가 되어 있는가?
