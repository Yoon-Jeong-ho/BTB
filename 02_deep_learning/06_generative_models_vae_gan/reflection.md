# 06 Generative Models: VAE, GAN 회고

## 1. VAE와 GAN을 내 말로 다시 구분하기
- VAE가 “latent를 샘플링 가능한 공간으로 정리한다”는 말을 이번 실습 숫자와 함께 다시 써 보자.
- GAN이 reconstruction 없이도 학습 신호를 만든다는 말을 adversarial 관점에서 설명해 보자.
- 두 모델이 모두 generative model이지만, 무엇을 우선시하는지 한 문장씩 비교해 보자.

## 2. latent sampling과 reparameterization trick 정리
- `mu`, `logvar`, `epsilon`, `z` 중 각각이 어떤 역할을 했는지 적어 보자.
- reparameterization trick이 왜 필요한지, 그냥 noise를 더하는 것과 무엇이 다른지 설명해 보자.
- prior sample spread나 interpolation path를 보고, latent geometry가 있다는 말을 어떻게 확인할 수 있었는가?

## 3. collapse failure를 분리해서 보기
- posterior collapse가 일어나면 어떤 지표와 샘플 관찰이 같이 무너졌는가?
- mode collapse가 일어나면 어떤 지표가 가장 먼저 경고를 주는가?
- “loss가 그럴듯하다”와 “다양한 샘플을 만든다”가 다른 말이라는 점을 이번 실습에서 어떻게 느꼈는가?

## 4. 다음 단위와 연결하기
- training recipe/debugging 관점에서 VAE와 GAN은 각각 어떤 모니터링 습관이 더 중요해 보이는가?
- 앞으로 diffusion model이나 multimodal generation을 볼 때도, latent structure / sample realism / collapse 위험이라는 세 축으로 어떻게 질문을 이어 갈 것인가?
