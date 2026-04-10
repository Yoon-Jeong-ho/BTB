# 05 Autoencoders and Representation Learning 선행 개념

## 꼭 알고 오면 좋은 것
- encoder/decoder 같은 다층 신경망이 입력 shape를 어떻게 바꾸는지 읽는 감각
- activation, loss, gradient가 각각 어떤 역할을 하는지에 대한 기본 이해
- MSE나 BCE 같은 reconstruction loss를 언제 쓰는지에 대한 감각
- hidden representation이 downstream feature로 재사용될 수 있다는 직관
- 차원 축소가 항상 좋은 것은 아니며, 정보 손실과 압축 이득을 같이 봐야 한다는 점

## 먼저 다시 보면 좋은 단위
- [00_foundations/01_tensor_shapes](../../00_foundations/01_tensor_shapes/README.md) — 입력/출력/latent shape 읽기 복습
- [00_foundations/02_activation_and_loss](../../00_foundations/02_activation_and_loss/README.md) — reconstruction loss 해석의 기초
- [00_foundations/03_gradients_and_backpropagation](../../00_foundations/03_gradients_and_backpropagation/README.md) — encoder/decoder 전체를 통한 gradient 흐름 복습
- [02_deep_learning/01_perceptron_and_mlp](../01_perceptron_and_mlp/README.md) — hidden representation과 nonlinearity를 다시 연결
- [02_deep_learning/02_cnn_and_image_classification](../02_cnn_and_image_classification/README.md) — 이미지 feature extractor 감각이 있다면 convolutional autoencoder로 이어 보기 좋음

## 빠른 자기 점검
- 입력 `x`, latent `z`, reconstruction `x_hat` 세 가지를 서로 다른 역할로 설명할 수 있는가?
- reconstruction loss가 label 없이도 학습 신호가 되는 이유를 한두 문장으로 말할 수 있는가?
- latent dimension을 줄이면 무엇을 얻고 무엇을 잃을 수 있는지 설명할 수 있는가?
- noisy input으로부터 clean target을 복원하게 하는 설정이 왜 representation을 더 강하게 만들 수 있는지 이해하는가?
- 복원 오차가 낮더라도 latent가 downstream task에서 항상 유용한 것은 아니라는 말을 받아들일 수 있는가?

## 이번 runnable 실습에 들어가기 전 팁
- scratch에서는 basis projection을 encoder/decoder 역할로 읽고, framework에서는 같은 아이디어가 PyTorch autoencoder 학습으로 어떻게 다시 나타나는지 본다.
- noisy input과 clean target을 따로 두는 이유를 먼저 이해하면 denoising variant를 훨씬 덜 헷갈린다.
- 숫자를 외우기보다 `latent dimension이 줄면 무엇을 잃는가?` 와 `denoising은 무엇을 지우는가?` 를 먼저 질문해도 충분하다.
