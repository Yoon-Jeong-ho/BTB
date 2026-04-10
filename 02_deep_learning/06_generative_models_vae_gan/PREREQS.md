# 06 Generative Models: VAE, GAN 선행 개념

## 꼭 알고 오면 좋은 것
- encoder, decoder, latent representation이 각각 어떤 역할을 하는지
- reconstruction loss가 label 없이도 학습 신호가 될 수 있다는 점
- 평균, 분산(또는 표준편차), 샘플링이라는 표현을 기본적으로 읽는 감각
- KL divergence를 “두 분포가 얼마나 다른가” 정도로 받아들이는 직관
- generator / discriminator가 동시에 학습되면 왜 해석이 더 까다로워지는지에 대한 감각
- sample quality와 diversity를 함께 봐야 한다는 평가 감각

## 먼저 다시 보면 좋은 단위
- [02_deep_learning/05_autoencoders_and_representation_learning](../05_autoencoders_and_representation_learning/README.md) — latent representation과 reconstruction objective 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — loss가 학습 신호를 주는 구조 복습
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — sampling이 있어도 결국 gradient로 학습된다는 점 복습
- [00_foundations/04_regularization_and_normalization](../../00_foundations/04_regularization_and_normalization/README.md) — KL/regularization과 학습 안정성 연결 복습
- [02_deep_learning/07_training_recipes_and_debugging](../07_training_recipes_and_debugging/README.md) — collapse와 imbalance를 나중에 어떻게 다룰지 미리 연결하기

## 빠른 자기 점검
- deterministic autoencoder와 샘플링 가능한 latent model의 차이를 한두 문장으로 말할 수 있는가?
- `mu`, `logvar`, `epsilon`, `z`를 봤을 때 각각이 무엇을 뜻하는지 읽을 수 있는가?
- KL divergence가 prior 쪽으로 latent를 정리하려는 압력이라는 직관이 있는가?
- adversarial training에서 loss 하나만으로 학습 상태를 다 읽기 어렵다는 점을 받아들일 수 있는가?
- posterior collapse와 mode collapse가 서로 다른 실패라는 점을 구분할 준비가 되어 있는가?
