# 02 Deep Learning

이 트랙은 `00_foundations`와 `01_ml` 다음에 놓이는 **딥러닝 모델 패밀리 학습** 구간이다. 앞선 두 트랙에서 텐서/gradient 감각과 고전 ML baseline 운영법을 익혔다면, 이제는 perceptron부터 transformer·generative model까지 `어떤 신경망 가족이 어떤 문제를 풀기 위해 등장했는지`를 계통적으로 연결한다.

즉 `00 → 01 → 02` 구간에서 공통 수학/실험 습관을 실제 neural architecture 감각으로 바꾸고, 이후 `03_nlp_bridge`, `04_nlp`, `05_advanced_nlp_llm`으로 올라가기 전에 모델 내부 구조를 한 번 정리하는 역할을 맡는다.

## 단위 구성

| Unit | Status | Focus |
| --- | --- | --- |
| [01_perceptron_and_mlp](01_perceptron_and_mlp/README.md) | outlined | 가장 작은 supervised neural model에서 출발해 hidden layer가 표현력을 어떻게 늘리는지 연결한다. |
| [02_cnn_and_image_classification](02_cnn_and_image_classification/README.md) | outlined | local receptive field, convolution, pooling이 이미지 분류에 왜 맞는지 본다. |
| [03_sequence_models_rnn_lstm_gru](03_sequence_models_rnn_lstm_gru/README.md) | outlined | 순서 정보와 hidden state 누적이 시퀀스 문제를 어떻게 다루는지 익힌다. |
| [04_attention_and_transformers](04_attention_and_transformers/README.md) | outlined | attention과 transformer가 장거리 의존성과 병렬 학습을 어떻게 바꿨는지 본다. |
| [05_autoencoders_and_representation_learning](05_autoencoders_and_representation_learning/README.md) | outlined | 복원 과제로 latent representation을 만드는 기본기를 정리한다. |
| [06_generative_models_vae_gan](06_generative_models_vae_gan/README.md) | planned | 확률적 생성과 adversarial 학습이라는 두 generative family를 비교한다. |
| [07_training_recipes_and_debugging](07_training_recipes_and_debugging/README.md) | outlined | 딥러닝 실험이 실제로 수렴하도록 만드는 운영 규칙과 failure triage를 정리한다. |

## 이 트랙에 포함되는 것

- perceptron, MLP, CNN, sequence model, attention/transformer, autoencoder, VAE/GAN 같은 대표 모델 계열
- representation learning과 latent space를 읽는 최소 실험 설계
- training recipe, failure pattern, debugging 관점의 기본기

## 이 트랙에서 아직 다루지 않는 것

- 대규모 분산 학습/병렬화 시스템 설계는 `06_training_systems`에서 다룬다.
- instruction tuning, preference optimization, RLHF 같은 post-pretraining LLM 정렬은 `05_advanced_nlp_llm`에서 다룬다.
- 장기 연구형 capstone과 agentic sandbox는 `07_frontier_labs`에서 다룬다.
